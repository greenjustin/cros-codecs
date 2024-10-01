// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Utility functions used by several parts of this crate.
//!
//! This module is for anything that doesn't fit into the other top-level modules. Try not to add
//! new code here unless it really doesn't belong anywhere else.

use std::borrow::Cow;
use std::io::Cursor;
use std::io::Seek;
use std::io::Write;
use std::marker::PhantomData;
use std::os::fd::OwnedFd;

use anyhow::anyhow;
use anyhow::Context;
use byteorder::ReadBytesExt;
use byteorder::LE;
use bytes::Buf;
use thiserror::Error;

use crate::codec::h264::parser::Nalu as H264Nalu;
use crate::codec::h265::parser::Nalu as H265Nalu;
use crate::decoder::stateless::DecodeError;
use crate::decoder::stateless::PoolLayer;
use crate::decoder::stateless::StatelessVideoDecoder;
use crate::decoder::BlockingMode;
use crate::decoder::DecodedHandle;
use crate::decoder::DecoderEvent;
use crate::decoder::FramePool;
use crate::decoder::StreamInfo;
use crate::DecodedFormat;
use crate::Fourcc;
use crate::FrameLayout;
use crate::PlaneLayout;
use crate::Resolution;

/// A bit reader for h264 bitstreams. It properly handles emulation-prevention
/// bytes and stop bits.
pub(crate) struct NaluReader<'a> {
    /// A reference into the next unread byte in the stream.
    data: Cursor<&'a [u8]>,
    /// Contents of the current byte. First unread bit starting at position 8 -
    /// num_remaining_bits_in_curr_bytes.
    curr_byte: u8,
    /// Number of bits remaining in `curr_byte`
    num_remaining_bits_in_curr_byte: usize,
    /// Used in epb detection.
    prev_two_bytes: u16,
    /// Number of epbs (i.e. 0x000003) we found.
    num_epb: usize,
    /// Whether or not we need emulation prevention logic.
    needs_epb: bool,
    /// How many bits have been read so far.
    position: u64,
}

#[derive(Debug, Error)]
pub(crate) enum GetByteError {
    #[error("reader ran out of bits")]
    OutOfBits,
}

#[derive(Debug, Error)]
pub(crate) enum ReadBitsError {
    #[error("more than 31 ({0}) bits were requested")]
    TooManyBytesRequested(usize),
    #[error("failed to advance the current byte")]
    GetByte(#[from] GetByteError),
    #[error("failed to convert read input to target type")]
    ConversionFailed,
}

impl<'a> NaluReader<'a> {
    pub fn new(data: &'a [u8], needs_epb: bool) -> Self {
        Self {
            data: Cursor::new(data),
            curr_byte: Default::default(),
            num_remaining_bits_in_curr_byte: Default::default(),
            prev_two_bytes: 0xffff,
            num_epb: Default::default(),
            needs_epb: needs_epb,
            position: 0,
        }
    }

    /// Read a single bit from the stream.
    pub fn read_bit(&mut self) -> Result<bool, ReadBitsError> {
        let bit = self.read_bits::<u32>(1)?;
        match bit {
            1 => Ok(true),
            0 => Ok(false),
            _ => panic!("Unexpected value {}", bit),
        }
    }

    /// Read up to 31 bits from the stream.
    pub fn read_bits<U: TryFrom<u32>>(&mut self, num_bits: usize) -> Result<U, ReadBitsError> {
        if num_bits > 31 {
            return Err(ReadBitsError::TooManyBytesRequested(num_bits));
        }

        let mut bits_left = num_bits;
        let mut out = 0u32;

        while self.num_remaining_bits_in_curr_byte < bits_left {
            out |= (self.curr_byte as u32) << (bits_left - self.num_remaining_bits_in_curr_byte);
            bits_left -= self.num_remaining_bits_in_curr_byte;
            self.update_curr_byte()?;
        }

        out |= (self.curr_byte >> (self.num_remaining_bits_in_curr_byte - bits_left)) as u32;
        out &= (1 << num_bits) - 1;
        self.num_remaining_bits_in_curr_byte -= bits_left;
        self.position += num_bits as u64;

        U::try_from(out).map_err(|_| ReadBitsError::ConversionFailed)
    }

    /// Skip `num_bits` bits from the stream.
    pub fn skip_bits(&mut self, mut num_bits: usize) -> Result<(), ReadBitsError> {
        while num_bits > 0 {
            let n = std::cmp::min(num_bits, 31);
            self.read_bits::<u32>(n)?;
            num_bits -= n;
        }

        Ok(())
    }

    /// Returns the amount of bits left in the stream
    pub fn num_bits_left(&self) -> usize {
        self.data.remaining() * 8 + self.num_remaining_bits_in_curr_byte
    }

    /// Returns the number of emulation-prevention bytes read so far.
    pub fn num_epb(&self) -> usize {
        self.num_epb
    }

    /// Whether the stream still has RBSP data. Implements more_rbsp_data(). See
    /// the spec for more details.
    pub fn has_more_rsbp_data(&mut self) -> bool {
        if self.num_remaining_bits_in_curr_byte == 0 && self.update_curr_byte().is_err() {
            // no more data at all in the rbsp
            return false;
        }

        // If the next bit is the stop bit, then we should only see unset bits
        // until the end of the data.
        if (self.curr_byte & ((1 << (self.num_remaining_bits_in_curr_byte - 1)) - 1)) != 0 {
            return true;
        }

        let data = self.data.chunk();
        for data in &data[0..self.data.remaining()] {
            if *data != 0 {
                return true;
            }
        }

        self.data.advance(self.data.remaining());
        false
    }

    pub fn read_ue<U: TryFrom<u32>>(&mut self) -> anyhow::Result<U> {
        let mut num_bits = 0;

        while self.read_bits::<u32>(1)? == 0 {
            num_bits += 1;
        }

        if num_bits > 31 {
            return Err(anyhow!("invalid stream"));
        }

        let value = ((1u32 << num_bits) - 1)
            .checked_add(self.read_bits::<u32>(num_bits)?)
            .context("read number cannot fit in 32 bits")?;

        U::try_from(value).map_err(|_| anyhow!("conversion error"))
    }

    pub fn read_ue_bounded<U: TryFrom<u32>>(&mut self, min: u32, max: u32) -> anyhow::Result<U> {
        let ue = self.read_ue()?;
        if ue > max || ue < min {
            Err(anyhow!(
                "Value out of bounds: expected {} - {}, got {}",
                min,
                max,
                ue
            ))
        } else {
            Ok(U::try_from(ue).map_err(|_| anyhow!("Conversion error"))?)
        }
    }

    pub fn read_ue_max<U: TryFrom<u32>>(&mut self, max: u32) -> anyhow::Result<U> {
        self.read_ue_bounded(0, max)
    }

    pub fn read_se<U: TryFrom<i32>>(&mut self) -> anyhow::Result<U> {
        let ue = self.read_ue::<u32>()? as i32;

        if ue % 2 == 0 {
            Ok(U::try_from(-ue / 2).map_err(|_| anyhow!("Conversion error"))?)
        } else {
            Ok(U::try_from(ue / 2 + 1).map_err(|_| anyhow!("Conversion error"))?)
        }
    }

    pub fn read_se_bounded<U: TryFrom<i32>>(&mut self, min: i32, max: i32) -> anyhow::Result<U> {
        let se = self.read_se()?;
        if se < min || se > max {
            Err(anyhow!(
                "Value out of bounds, expected between {}-{}, got {}",
                min,
                max,
                se
            ))
        } else {
            Ok(U::try_from(se).map_err(|_| anyhow!("Conversion error"))?)
        }
    }

    pub fn read_le<U: TryFrom<u32>>(&mut self, num_bits: u8) -> anyhow::Result<U> {
        if self.num_remaining_bits_in_curr_byte % 8 != 0 {
            return Err(anyhow!("Attempted unaligned read_le()"));
        }

        let mut t = 0;

        for i in 0..num_bits {
            let byte = self.read_bits::<u32>(8)?;
            t += byte << (i * 8)
        }

        Ok(U::try_from(t).map_err(|_| anyhow!("Conversion error"))?)
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    fn get_byte(&mut self) -> Result<u8, GetByteError> {
        self.data.read_u8().map_err(|_| GetByteError::OutOfBits)
    }

    fn update_curr_byte(&mut self) -> Result<(), GetByteError> {
        let mut byte = self.get_byte()?;

        if self.needs_epb {
            if self.prev_two_bytes == 0 && byte == 0x03 {
                // We found an epb
                self.num_epb += 1;
                // Read another byte
                byte = self.get_byte()?;
                // We need another 3 bytes before another epb can happen.
                self.prev_two_bytes = 0xffff;
            }
            self.prev_two_bytes = (self.prev_two_bytes << 8) | u16::from(byte);
        }

        self.num_remaining_bits_in_curr_byte = 8;
        self.curr_byte = byte;
        Ok(())
    }
}

/// Iterator over IVF packets.
pub struct IvfIterator<'a> {
    cursor: Cursor<&'a [u8]>,
}

impl<'a> IvfIterator<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut cursor = Cursor::new(data);

        // Skip the IVH header entirely.
        cursor.seek(std::io::SeekFrom::Start(32)).unwrap();

        Self { cursor }
    }
}

impl<'a> Iterator for IvfIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Make sure we have a header.
        if self.cursor.remaining() < 12 {
            return None;
        }

        let len = self.cursor.read_u32::<LE>().ok()? as usize;
        // Skip PTS.
        let _ = self.cursor.read_u64::<LE>().ok()?;

        if self.cursor.remaining() < len {
            return None;
        }

        let start = self.cursor.position() as usize;
        let _ = self.cursor.seek(std::io::SeekFrom::Current(len as i64));
        let end = self.cursor.position() as usize;

        Some(&self.cursor.get_ref()[start..end])
    }
}

/// Helper struct for synthesizing IVF file header
pub struct IvfFileHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub header_size: u16,
    pub codec: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub framerate: u32,
    pub timescale: u32,
    pub frame_count: u32,
    pub unused: u32,
}

impl Default for IvfFileHeader {
    fn default() -> Self {
        Self {
            magic: Self::MAGIC,
            version: 0,
            header_size: 32,
            codec: Self::CODEC_VP9,
            width: 320,
            height: 240,
            framerate: 1,
            timescale: 1000,
            frame_count: 1,
            unused: Default::default(),
        }
    }
}

impl IvfFileHeader {
    pub const MAGIC: [u8; 4] = *b"DKIF";
    pub const CODEC_VP8: [u8; 4] = *b"VP80";
    pub const CODEC_VP9: [u8; 4] = *b"VP90";
    pub const CODEC_AV1: [u8; 4] = *b"AV01";

    pub fn new(codec: [u8; 4], width: u16, height: u16, framerate: u32, frame_count: u32) -> Self {
        let default = Self::default();

        Self {
            codec,
            width,
            height,
            framerate: framerate * default.timescale,
            frame_count,
            ..default
        }
    }
}

impl IvfFileHeader {
    /// Writes header into writer
    pub fn writo_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.header_size.to_le_bytes())?;
        writer.write_all(&self.codec)?;
        writer.write_all(&self.width.to_le_bytes())?;
        writer.write_all(&self.height.to_le_bytes())?;
        writer.write_all(&self.framerate.to_le_bytes())?;
        writer.write_all(&self.timescale.to_le_bytes())?;
        writer.write_all(&self.frame_count.to_le_bytes())?;
        writer.write_all(&self.unused.to_le_bytes())?;

        Ok(())
    }
}

/// Helper struct for synthesizing IVF frame header
pub struct IvfFrameHeader {
    pub frame_size: u32,
    pub timestamp: u64,
}

impl IvfFrameHeader {
    /// Writes header into writer
    pub fn writo_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writer.write_all(&self.frame_size.to_le_bytes())?;
        writer.write_all(&self.timestamp.to_le_bytes())?;
        Ok(())
    }
}

/// Iterator NALUs in a bitstream.
pub struct NalIterator<'a, Nalu>(Cursor<&'a [u8]>, PhantomData<Nalu>);

impl<'a, Nalu> NalIterator<'a, Nalu> {
    pub fn new(stream: &'a [u8]) -> Self {
        Self(Cursor::new(stream), PhantomData)
    }
}

impl<'a> Iterator for NalIterator<'a, H264Nalu<'a>> {
    type Item = Cow<'a, [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        H264Nalu::next(&mut self.0).map(|n| n.data).ok()
    }
}

impl<'a> Iterator for NalIterator<'a, H265Nalu<'a>> {
    type Item = Cow<'a, [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        H265Nalu::next(&mut self.0).map(|n| n.data).ok()
    }
}

#[derive(Error, Debug)]
pub enum BitWriterError {
    #[error("invalid bit count")]
    InvalidBitCount,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type BitWriterResult<T> = std::result::Result<T, BitWriterError>;

pub struct BitWriter<W: Write> {
    out: W,
    nth_bit: u8,
    curr_byte: u8,
}

impl<W: Write> BitWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            out: writer,
            curr_byte: 0,
            nth_bit: 0,
        }
    }

    /// Writes fixed bit size integer (up to 32 bit)
    pub fn write_f<T: Into<u32>>(&mut self, bits: usize, value: T) -> BitWriterResult<usize> {
        let value = value.into();

        if bits > 32 {
            return Err(BitWriterError::InvalidBitCount);
        }

        let mut written = 0;
        for bit in (0..bits).rev() {
            let bit = (1 << bit) as u32;

            self.write_bit((value & bit) == bit)?;
            written += 1;
        }

        Ok(written)
    }

    /// Takes a single bit that will be outputed to [`std::io::Write`]
    pub fn write_bit(&mut self, bit: bool) -> BitWriterResult<()> {
        self.curr_byte |= (bit as u8) << (7u8 - self.nth_bit);
        self.nth_bit += 1;

        if self.nth_bit == 8 {
            self.out.write_all(&[self.curr_byte])?;
            self.nth_bit = 0;
            self.curr_byte = 0;
        }

        Ok(())
    }

    /// Immediately outputs any cached bits to [`std::io::Write`]
    pub fn flush(&mut self) -> BitWriterResult<()> {
        if self.nth_bit != 0 {
            self.out.write_all(&[self.curr_byte])?;
            self.nth_bit = 0;
            self.curr_byte = 0;
        }

        self.out.flush()?;
        Ok(())
    }

    /// Returns `true` if ['Self`] hold data that wasn't written to [`std::io::Write`]
    pub fn has_data_pending(&self) -> bool {
        self.nth_bit != 0
    }

    pub(crate) fn inner(&self) -> &W {
        &self.out
    }

    pub(crate) fn inner_mut(&mut self) -> &mut W {
        &mut self.out
    }
}

impl<W: Write> Drop for BitWriter<W> {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            log::error!("Unable to flush bits {e:?}");
        }
    }
}

/// Simple decoding loop that plays the stream once from start to finish.
#[allow(clippy::type_complexity)]
pub fn simple_playback_loop<D, R, I, H, FP>(
    decoder: &mut D,
    stream_iter: I,
    on_new_frame: &mut dyn FnMut(H),
    allocate_new_frames: &mut dyn FnMut(&StreamInfo, usize) -> anyhow::Result<Vec<H::Descriptor>>,
    output_format: DecodedFormat,
    blocking_mode: BlockingMode,
) -> anyhow::Result<()>
where
    H: DecodedHandle,
    FP: FramePool<Descriptor = H::Descriptor> + ?Sized,
    D: StatelessVideoDecoder<Handle = H, FramePool = FP> + ?Sized,
    R: AsRef<[u8]>,
    I: Iterator<Item = R>,
{
    // Closure that drains all pending decoder events and calls `on_new_frame` on each
    // completed frame.
    let mut check_events = |decoder: &mut D| -> anyhow::Result<()> {
        while let Some(event) = decoder.next_event() {
            match event {
                DecoderEvent::FrameReady(frame) => {
                    on_new_frame(frame);
                }
                DecoderEvent::FormatChanged(mut format_setter) => {
                    format_setter.try_format(output_format).unwrap();
                    let stream_info = format_setter.stream_info().clone();
                    let min_num_frames = stream_info.min_num_frames;
                    /* we need to account for multiple layers if applicable for
                     * the stream */
                    let pools = format_setter.frame_pool(PoolLayer::All);
                    let nb_pools = pools.len();
                    for pool in pools {
                        // Allocate the missing number of buffers in our pool for decoding to succeed.
                        let pool_num_frames = pool.num_managed_frames();
                        if pool_num_frames < (min_num_frames / nb_pools) {
                            let frames = allocate_new_frames(
                                &stream_info,
                                min_num_frames - pool_num_frames,
                            )?;
                            pool.add_frames(frames).unwrap();
                        }
                    }
                }
            }
        }

        Ok(())
    };

    for (frame_num, packet) in stream_iter.enumerate() {
        let mut bitstream = packet.as_ref();
        loop {
            match decoder.decode(frame_num as u64, bitstream) {
                Ok(bytes_decoded) => {
                    bitstream = &bitstream[bytes_decoded..];

                    if blocking_mode == BlockingMode::Blocking {
                        check_events(decoder)?;
                    }

                    if bitstream.is_empty() {
                        // Break the loop so we can process the next NAL if we sent the current one
                        // successfully.
                        break;
                    }
                }
                Err(DecodeError::CheckEvents) | Err(DecodeError::NotEnoughOutputBuffers(_)) => {
                    check_events(decoder)?
                }
                Err(e) => anyhow::bail!(e),
            }
        }
    }

    decoder.flush()?;
    check_events(decoder)
}

/// Frame allocation callback that results in self-allocated memory.
pub fn simple_playback_loop_owned_frames(
    _: &StreamInfo,
    nb_frames: usize,
) -> anyhow::Result<Vec<()>> {
    Ok(vec![(); nb_frames])
}

/// Frame allocation callback that returns user-allocated memory for the frames.
pub fn simple_playback_loop_userptr_frames(
    stream_info: &StreamInfo,
    nb_frames: usize,
) -> anyhow::Result<Vec<UserPtrFrame>> {
    let alloc_function = match stream_info.format {
        DecodedFormat::I420 | DecodedFormat::NV12 => &UserPtrFrame::new_nv12,
        _ => anyhow::bail!(
            "{:?} format is unsupported with user memory",
            stream_info.format
        ),
    };

    Ok((0..nb_frames)
        .map(|_| alloc_function(stream_info.coded_resolution))
        .collect::<Vec<_>>())
}

/// A structure that holds user-allocated memory for a frame as well as its layout.
#[derive(Debug)]
pub struct UserPtrFrame {
    pub buffers: Vec<*mut u8>,
    pub mem_layout: std::alloc::Layout,
    pub layout: FrameLayout,
}

impl UserPtrFrame {
    /// Allocate enough memory to back a NV12 frame of `size` dimension.
    pub fn new_nv12(size: Resolution) -> Self {
        /// Add what is needed to a value in order to make it a multiple of some alignment.
        macro_rules! align {
            ($n:expr, $r:expr) => {
                ($n + ($r - 1)) & !($r - 1)
            };
        }

        // Align according to VAAPI constraints.
        let width = align!(size.width, 16) as usize;
        let height = align!(size.height, 4) as usize;
        let stride = align!(width, 64);
        let uv_start = height * stride;
        let uv_size = (height / 2) * stride;

        Self::alloc(
            FrameLayout {
                format: (Fourcc::from(b"NV12"), 0),
                size: Resolution::from((width as u32, height as u32)),
                planes: vec![
                    PlaneLayout {
                        buffer_index: 0,
                        offset: 0,
                        stride,
                    },
                    PlaneLayout {
                        buffer_index: 0,
                        offset: uv_start,
                        stride,
                    },
                ],
            },
            uv_start.max(uv_size),
        )
    }

    pub fn alloc(layout: FrameLayout, buffer_size: usize) -> Self {
        let buffer_count = layout
            .planes
            .iter()
            .map(|plane| plane.buffer_index)
            .collect::<std::collections::HashSet<usize>>()
            .len();

        // SAFETY: the invariants of `Layout` are respected.
        let mem_layout =
            unsafe { std::alloc::Layout::from_size_align_unchecked(buffer_size, 4096) };

        let buffers = (0..buffer_count)
            .map(|_| {
                // SAFETY: the invariants of `Layout` are respected.
                unsafe { std::alloc::alloc(mem_layout) }
            })
            .collect();

        Self {
            buffers,
            mem_layout,
            layout,
        }
    }
}

#[derive(Debug)]
pub struct DmabufFrame {
    pub fds: Vec<OwnedFd>,
    pub layout: FrameLayout,
}

impl Drop for UserPtrFrame {
    fn drop(&mut self) {
        for buffer in std::mem::take(&mut self.buffers).into_iter() {
            // Safe because we allocated the memory using `std::alloc::alloc`.
            unsafe { std::alloc::dealloc(buffer, self.mem_layout) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_file_header() {
        let mut hdr = IvfFileHeader {
            version: 0,
            codec: IvfFileHeader::CODEC_VP9,
            width: 256,
            height: 256,
            framerate: 30_000,
            timescale: 1_000,
            frame_count: 1,

            ..Default::default()
        };

        let mut buf = Vec::new();
        hdr.writo_into(&mut buf).unwrap();

        const EXPECTED: [u8; 32] = [
            0x44, 0x4b, 0x49, 0x46, 0x00, 0x00, 0x20, 0x00, 0x56, 0x50, 0x39, 0x30, 0x00, 0x01,
            0x00, 0x01, 0x30, 0x75, 0x00, 0x00, 0xe8, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];

        assert_eq!(&buf, &EXPECTED);

        hdr.width = 1920;
        hdr.height = 800;
        hdr.framerate = 24;
        hdr.timescale = 1;
        hdr.frame_count = 100;

        buf.clear();
        hdr.writo_into(&mut buf).unwrap();

        const EXPECTED2: [u8; 32] = [
            0x44, 0x4b, 0x49, 0x46, 0x00, 0x00, 0x20, 0x00, 0x56, 0x50, 0x39, 0x30, 0x80, 0x07,
            0x20, 0x03, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];

        assert_eq!(&buf, &EXPECTED2);
    }

    #[test]
    fn test_ivf_frame_header() {
        let mut hdr = IvfFrameHeader {
            frame_size: 199249,
            timestamp: 0,
        };

        let mut buf = Vec::new();
        hdr.writo_into(&mut buf).unwrap();

        const EXPECTED: [u8; 12] = [
            0x51, 0x0a, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        assert_eq!(&buf, &EXPECTED);

        hdr.timestamp = 1;
        hdr.frame_size = 52;

        buf.clear();
        hdr.writo_into(&mut buf).unwrap();

        const EXPECTED2: [u8; 12] = [
            0x34, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        assert_eq!(&buf, &EXPECTED2);
    }

    #[test]
    fn test_bitwriter_f1() {
        let mut buf = Vec::<u8>::new();
        {
            let mut writer = BitWriter::new(&mut buf);
            writer.write_f(1, true).unwrap();
            writer.write_f(1, false).unwrap();
            writer.write_f(1, false).unwrap();
            writer.write_f(1, false).unwrap();
            writer.write_f(1, true).unwrap();
            writer.write_f(1, true).unwrap();
            writer.write_f(1, true).unwrap();
            writer.write_f(1, true).unwrap();
        }
        assert_eq!(buf, vec![0b10001111u8]);
    }

    #[test]
    fn test_bitwriter_f3() {
        let mut buf = Vec::<u8>::new();
        {
            let mut writer = BitWriter::new(&mut buf);
            writer.write_f(3, 0b100u8).unwrap();
            writer.write_f(3, 0b101u8).unwrap();
            writer.write_f(3, 0b011u8).unwrap();
        }
        assert_eq!(buf, vec![0b10010101u8, 0b10000000u8]);
    }

    #[test]
    fn test_bitwriter_f4() {
        let mut buf = Vec::<u8>::new();
        {
            let mut writer = BitWriter::new(&mut buf);
            writer.write_f(4, 0b1000u8).unwrap();
            writer.write_f(4, 0b1011u8).unwrap();
        }
        assert_eq!(buf, vec![0b10001011u8]);
    }

    // These tests are adapted from the chromium tests at media/video/h264_bit_reader_unitttest.cc

    #[test]
    fn read_stream_without_escape_and_trailing_zero_bytes() {
        const RBSP: [u8; 6] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xa0];

        let mut reader = NaluReader::new(&RBSP, true);
        assert_eq!(reader.read_bits::<u32>(1).unwrap(), 0);
        assert_eq!(reader.num_bits_left(), 47);
        assert!(reader.has_more_rsbp_data());

        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x02);
        assert_eq!(reader.num_bits_left(), 39);
        assert!(reader.has_more_rsbp_data());

        assert_eq!(reader.read_bits::<u32>(31).unwrap(), 0x23456789);
        assert_eq!(reader.num_bits_left(), 8);
        assert!(reader.has_more_rsbp_data());

        assert_eq!(reader.read_bits::<u32>(1).unwrap(), 1);
        assert_eq!(reader.num_bits_left(), 7);
        assert!(reader.has_more_rsbp_data());

        assert_eq!(reader.read_bits::<u32>(1).unwrap(), 0);
        assert_eq!(reader.num_bits_left(), 6);
        assert!(!reader.has_more_rsbp_data());
    }

    #[test]
    fn single_byte_stream() {
        const RBSP: [u8; 1] = [0x18];

        let mut reader = NaluReader::new(&RBSP, true);
        assert_eq!(reader.num_bits_left(), 8);
        assert!(reader.has_more_rsbp_data());
        assert_eq!(reader.read_bits::<u32>(4).unwrap(), 1);
        assert!(!reader.has_more_rsbp_data());
    }

    #[test]
    fn stop_bit_occupy_full_byte() {
        const RBSP: [u8; 2] = [0xab, 0x80];

        let mut reader = NaluReader::new(&RBSP, true);
        assert_eq!(reader.num_bits_left(), 16);
        assert!(reader.has_more_rsbp_data());

        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0xab);
        assert_eq!(reader.num_bits_left(), 8);

        assert!(!reader.has_more_rsbp_data());
    }

    // Check that read_ue behaves properly with input at the limits.
    #[test]
    fn read_ue() {
        // Regular value.
        let mut reader = NaluReader::new(&[0b0001_1010], true);
        assert_eq!(reader.read_ue::<u32>().unwrap(), 12);
        assert_eq!(reader.data.position(), 1);
        assert_eq!(reader.num_remaining_bits_in_curr_byte, 1);

        // 0 value.
        let mut reader = NaluReader::new(&[0b1000_0000], true);
        assert_eq!(reader.read_ue::<u32>().unwrap(), 0);
        assert_eq!(reader.data.position(), 1);
        assert_eq!(reader.num_remaining_bits_in_curr_byte, 7);

        // No prefix stop bit.
        let mut reader = NaluReader::new(&[0b0000_0000], true);
        reader.read_ue::<u32>().unwrap_err();

        // u32 max value: 31 0-bits, 1 bit marker, 31 bits 1-bits.
        let mut reader = NaluReader::new(&[
            0b0000_0000,
            0b0000_0000,
            0b0000_0000,
            0b0000_0001,
            0b1111_1111,
            0b1111_1111,
            0b1111_1111,
            0b1111_1110,
        ], true);
        assert_eq!(reader.read_ue::<u32>().unwrap(), 0xffff_fffe);
        assert_eq!(reader.data.position(), 8);
        assert_eq!(reader.num_remaining_bits_in_curr_byte, 1);
    }

    // Check that emulation prevention is being handled correctly.
    #[test]
    fn skip_epb_when_enabled() {
        let mut reader = NaluReader::new(&[0x00, 0x00, 0x03, 0x01], false);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x00);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x00);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x03);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x01);

        let mut reader = NaluReader::new(&[0x00, 0x00, 0x03, 0x01], true);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x00);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x00);
        assert_eq!(reader.read_bits::<u32>(8).unwrap(), 0x01);
    }
}
