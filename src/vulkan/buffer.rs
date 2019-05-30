use super::{context::*, util::*};
use ash::{version::DeviceV1_0, vk};
use std::{mem::size_of, rc::Rc};

pub struct Buffer {
    context: Rc<Context>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Buffer {
    fn new(
        context: Rc<Context>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
    ) -> Self {
        Self {
            context,
            buffer,
            memory,
            size,
        }
    }

    /// Create a buffer and allocate its memory.
    ///
    /// # Returns
    ///
    /// The buffer, its memory and the actual size in bytes of the
    /// allocated memory since in may differ from the requested size.
    pub fn create(
        context: Rc<Context>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let device = context.device();
        let buffer = {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { device.create_buffer(&buffer_info, None).unwrap() }
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type = find_memory_type(
                mem_requirements,
                context.get_mem_properties(),
                mem_properties,
            );

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type);
            unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
        };

        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };

        Buffer::new(context, buffer, memory, mem_requirements.size)
    }
}

impl Buffer {
    /// Copy the `size` first bytes of `src` this buffer.
    ///
    /// It's done using a command buffer allocated from
    /// `command_pool`. The command buffer is cubmitted tp
    /// `transfer_queue`.
    pub fn copy(&self, src: &Buffer, size: vk::DeviceSize) {
        self.context.execute_one_time_commands(|buffer| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            let regions = [region];

            unsafe {
                self.context
                    .device()
                    .cmd_copy_buffer(buffer, src.buffer, self.buffer, &regions)
            };
        });
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_buffer(self.buffer, None);
            self.context.device().free_memory(self.memory, None);
        }
    }
}

/// Create a buffer and it's gpu  memory and fill it.
///
/// This function internally creates an host visible staging buffer and
/// a device local buffer. The data is first copied from the cpu to the
/// staging buffer. Then we copy the data from the staging buffer to the
/// final buffer using a one-time command buffer.
pub fn create_device_local_buffer_with_data<A, T: Copy>(
    context: &Rc<Context>,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> Buffer {
    let device = context.device();
    let size = (data.len() * size_of::<T>()) as vk::DeviceSize;
    let staging_buffer = Buffer::create(
        Rc::clone(context),
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    unsafe {
        let data_ptr = device
            .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        mem_copy(data_ptr, data);
        device.unmap_memory(staging_buffer.memory);
    };

    let buffer = Buffer::create(
        Rc::clone(context),
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    buffer.copy(&staging_buffer, staging_buffer.size);

    buffer
}
