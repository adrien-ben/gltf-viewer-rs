use ash::extensions::ext::DebugReport;
use ash::{vk, Entry, Instance};
use std::{
    ffi::CStr,
    os::raw::{c_char, c_void},
};

#[cfg(debug_assertions)]
pub const ENABLE_DEBUG_CALLBACK: bool = true;
#[cfg(not(debug_assertions))]
pub const ENABLE_DEBUG_CALLBACK: bool = false;

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugReportFlagsEXT,
    typ: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    if flag == vk::DebugReportFlagsEXT::DEBUG {
        log::debug!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::INFORMATION {
        log::info!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::PERFORMANCE_WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else {
        log::error!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    }
    vk::FALSE
}

/// Setup the debug message if validation layers are enabled.
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(DebugReport, vk::DebugReportCallbackEXT)> {
    if !ENABLE_DEBUG_CALLBACK {
        return None;
    }
    let create_info = vk::DebugReportCallbackCreateInfoEXT::builder()
        .flags(vk::DebugReportFlagsEXT::all())
        .pfn_callback(Some(vulkan_debug_callback));
    let debug_report = DebugReport::new(entry, instance);
    let debug_report_callback = unsafe {
        debug_report
            .create_debug_report_callback(&create_info, None)
            .expect("Failed to create debig report callback")
    };
    Some((debug_report, debug_report_callback))
}
