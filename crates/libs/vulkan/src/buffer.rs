use super::{context::*, util::*};
use ash::vk;
use std::{
    ffi::c_void,
    marker::{Send, Sync},
    mem::size_of_val,
    sync::Arc,
};

/// Wrapper over a raw pointer to make it moveable and accessible from other threads
struct MemoryMapPointer(*mut c_void);
unsafe impl Send for MemoryMapPointer {}
unsafe impl Sync for MemoryMapPointer {}

pub struct Buffer {
    context: Arc<Context>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    mapped_pointer: Option<MemoryMapPointer>,
}

impl Buffer {
    fn new(
        context: Arc<Context>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
    ) -> Self {
        Self {
            context,
            buffer,
            memory,
            size,
            mapped_pointer: None,
        }
    }

    /// Create a buffer and allocate its memory.
    ///
    /// # Returns
    ///
    /// The buffer, its memory and the actual size in bytes of the
    /// allocated memory since in may differ from the requested size.
    pub fn create(
        context: Arc<Context>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let device = context.device();
        let buffer = {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe {
                device
                    .create_buffer(&buffer_info, None)
                    .expect("Failed to create buffer")
            }
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type = find_memory_type(
                mem_requirements,
                context.get_mem_properties(),
                mem_properties,
            );

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type);
            unsafe {
                device
                    .allocate_memory(&alloc_info, None)
                    .expect("Failed to allocate memory")
            }
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind buffer memory")
        };

        Buffer::new(context, buffer, memory, size)
    }
}

impl Buffer {
    /// Register the commands to copy the `size` first bytes of `src` this buffer.
    pub fn cmd_copy(&self, command_buffer: vk::CommandBuffer, src: &Buffer, size: vk::DeviceSize) {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        let regions = [region];

        unsafe {
            self.context
                .device()
                .cmd_copy_buffer(command_buffer, src.buffer, self.buffer, &regions)
        };
    }

    /// Map the buffer memory and return the mapped pointer.
    ///
    /// If the memory is already mapped it just returns the pointer.
    pub fn map_memory(&mut self) -> *mut c_void {
        if let Some(ptr) = &self.mapped_pointer {
            ptr.0
        } else {
            unsafe {
                let ptr = self
                    .context
                    .device()
                    .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                    .expect("Failed to map memory");
                self.mapped_pointer = Some(MemoryMapPointer(ptr));
                ptr
            }
        }
    }

    /// Unmap the buffer memory.
    ///
    /// Does nothing if memory is not mapped.
    pub fn unmap_memory(&mut self) {
        if self.mapped_pointer.take().is_some() {
            unsafe {
                self.context.device().unmap_memory(self.memory);
            }
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.unmap_memory();
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
    context: &Arc<Context>,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> Buffer {
    let (buffer, _) = context.execute_one_time_commands(|command_buffer| {
        cmd_create_device_local_buffer_with_data::<A, _>(context, command_buffer, usage, data)
    });
    buffer
}

pub fn cmd_create_device_local_buffer_with_data<A, T: Copy>(
    context: &Arc<Context>,
    command_buffer: vk::CommandBuffer,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> (Buffer, Buffer) {
    let size = size_of_val(data) as vk::DeviceSize;
    let staging_buffer =
        create_host_visible_buffer(context, vk::BufferUsageFlags::TRANSFER_SRC, data);
    let buffer = Buffer::create(
        Arc::clone(context),
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    buffer.cmd_copy(command_buffer, &staging_buffer, staging_buffer.size);

    (buffer, staging_buffer)
}

pub fn create_host_visible_buffer<T: Copy>(
    context: &Arc<Context>,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> Buffer {
    let size = size_of_val(data) as vk::DeviceSize;
    let mut buffer = Buffer::create(
        Arc::clone(context),
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    unsafe {
        let data_ptr = buffer.map_memory();
        mem_copy(data_ptr, data);
    };

    buffer
}
