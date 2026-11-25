//! Lightweight declarative Render Graph with automatic `Synchronization2` layout
//! and stage/access barrier insertion.

use std::collections::HashMap;

use engine_rhi::ash::vk;
use engine_rhi::sync;

/// Opaque handle to an image resource tracked by [`RenderGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RgImageId(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct RgImageState {
    pub image: vk::Image,
    pub range: vk::ImageSubresourceRange,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Debug, Clone, Copy)]
pub struct RgAccess {
    pub resource: RgImageId,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

/// Declarative pass DAG & barrier compiler.
#[derive(Default)]
pub struct RenderGraph {
    resources: Vec<RgImageState>,
    name_to_id: HashMap<&'static str, RgImageId>,
    passes: Vec<(&'static str, Vec<RgAccess>)>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_image(
        &mut self,
        name: &'static str,
        image: vk::Image,
        range: vk::ImageSubresourceRange,
        initial_layout: vk::ImageLayout,
    ) -> RgImageId {
        let id = RgImageId(self.resources.len());
        self.resources.push(RgImageState {
            image,
            range,
            layout: initial_layout,
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access: vk::AccessFlags2::NONE,
        });
        self.name_to_id.insert(name, id);
        id
    }

    pub fn add_pass(&mut self, name: &'static str, accesses: Vec<RgAccess>) {
        self.passes.push((name, accesses));
    }

    /// Emit `Synchronization2` barriers for a single pass and update tracked resource states.
    pub fn emit_pass_barriers(
        &mut self,
        device: &engine_rhi::ash::Device,
        cmd: vk::CommandBuffer,
        accesses: &[RgAccess],
    ) {
        let mut barriers = Vec::with_capacity(accesses.len());
        for req in accesses {
            let st = &mut self.resources[req.resource.0];
            if st.layout != req.layout || st.access != vk::AccessFlags2::NONE {
                barriers.push(sync::image_barrier(
                    st.image,
                    st.layout,
                    req.layout,
                    st.stage,
                    st.access,
                    req.stage,
                    req.access,
                    st.range,
                ));
            }
            st.layout = req.layout;
            st.stage = req.stage;
            st.access = req.access;
        }
        if !barriers.is_empty() {
            unsafe {
                sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &barriers);
            }
        }
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|(n, _)| *n).collect()
    }
}
