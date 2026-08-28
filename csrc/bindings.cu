#include "kimi_k3_decode/entrypoints.cuh"
#include "kimi_k3_decode/expert_mxfp4.cuh"
#include "megakernel/entrypoints.cuh"
#include "mxfp8.cuh"
#include "scheduler.cuh"
#include "utils.cuh"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kimi_k3_decode", &kimi_k3_decode::kimi_k3_decode_entrypoint, "",
          pybind11::arg("hidden_states"),
          pybind11::arg("router_weight"), pybind11::arg("router_correction_bias"),
          pybind11::arg("routed_expert_down_proj"), pybind11::arg("routed_expert_up_proj"),
          pybind11::arg("routed_latent_rmsnorm_weight"),
          pybind11::arg("expert_w1_packed"), pybind11::arg("expert_w1_scale"),
          pybind11::arg("expert_w3_packed"), pybind11::arg("expert_w3_scale"),
          pybind11::arg("expert_w2_packed"), pybind11::arg("expert_w2_scale"),
          pybind11::arg("shared_gate_proj"), pybind11::arg("shared_up_proj"),
          pybind11::arg("shared_down_proj"),
          pybind11::arg("scratch"),
          pybind11::arg("collective_buffer"), pybind11::arg("collective_buffer_ptrs"),
          pybind11::arg("collective_buffer_multicast_ptr"),
          pybind11::arg("output_mailbox"), pybind11::arg("output_mailbox_ptrs"),
          pybind11::arg("output_mailbox_multicast_ptr"),
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"),
          pybind11::arg("error_flag"),
          pybind11::arg("tp_rank"), pybind11::arg("active_tokens"),
          pybind11::arg("workspace_signature"));
    m.def("kimi_k3_decode_workspace_bytes",
          &kimi_k3_decode::kimi_k3_decode_workspace_bytes);
    m.def("_kimi_k3_decode_task_plan",
          &kimi_k3_decode::persistent::task_plan_for_testing, "",
          pybind11::arg("active_tokens"));
    m.def("_kimi_k3_decode_validate_residency",
          &kimi_k3_decode::persistent::validate_residency, "",
          pybind11::arg("available_sms"),
          pybind11::arg("blocks_per_sm"));
    m.def("_kimi_k3_decode_resident_blocks_per_sm",
          &kimi_k3_decode::persistent::resident_blocks_per_sm_for_testing, "",
          pybind11::arg("tensor_path"));
    m.def("_kimi_k3_decode_shared_memory_reservations",
          &kimi_k3_decode::persistent::shared_memory_reservations_for_testing,
          "", pybind11::arg("device"));
    m.def("_kimi_k3_decode_timeout_metadata",
          &kimi_k3_decode::persistent::timeout_metadata_for_testing);
    m.def("_kimi_k3_decode_queue_bound",
          &kimi_k3_decode::persistent::queue_bound_for_testing);
    m.def("_kimi_k3_decode_wait_timeout_clocks", []() {
        return kimi_k3_decode::persistent::kWaitTimeoutClocks;
    });
    m.def("_kimi_k3_timeout_sites", []() {
        std::vector<std::tuple<std::string, std::int64_t, std::int64_t,
                               std::int64_t>> sites;
        for (const auto &site : kimi_k3_decode::kTimeoutSites) {
            sites.emplace_back(
                site.name, static_cast<std::int64_t>(site.code),
                static_cast<std::int64_t>(site.timeout_slot),
                static_cast<std::int64_t>(site.counter));
        }
        return sites;
    });
    m.def("_kimi_k3_decode_grid_shape", []() {
        return std::make_tuple(
            static_cast<std::int64_t>(kimi_k3_decode::persistent::kPersistentCtas),
            static_cast<std::int64_t>(kimi_k3_decode::kDecodeCtaThreads),
            static_cast<std::int64_t>(
                kimi_k3_decode::persistent::kPersistentSharedBytes));
    });
    m.def("_kimi_k3_route_and_project",
          &kimi_k3_decode::route_and_project_entrypoint, "",
          pybind11::arg("hidden_states"),
          pybind11::arg("router_weight"), pybind11::arg("router_correction_bias"),
          pybind11::arg("routed_expert_down_proj"),
          pybind11::arg("scratch"), pybind11::arg("active_tokens"));
    m.def("_kimi_k3_routed_experts",
          &kimi_k3_decode::routed_experts_entrypoint, "",
          pybind11::arg("latent_x"),
          pybind11::arg("expert_w1_packed"),
          pybind11::arg("expert_w1_scale"),
          pybind11::arg("expert_w3_packed"),
          pybind11::arg("expert_w3_scale"),
          pybind11::arg("expert_w2_packed"),
          pybind11::arg("expert_w2_scale"),
          pybind11::arg("routed_output"),
          pybind11::arg("scratch"),
          pybind11::arg("active_tokens"));
    m.def("_kimi_k3_shared_experts",
          &kimi_k3_decode::shared_experts_entrypoint, "",
          pybind11::arg("hidden_states"),
          pybind11::arg("shared_gate_proj"),
          pybind11::arg("shared_up_proj"),
          pybind11::arg("shared_down_proj"),
          pybind11::arg("scratch"),
          pybind11::arg("collective_buffer"),
          pybind11::arg("active_tokens"));
    m.def("_kimi_k3_shared_experts_role_plan",
          &kimi_k3_decode::shared_experts::role_plan_for_testing, "",
          pybind11::arg("active_tokens"));
    m.def("_kimi_k3_shared_experts_validate_residency",
          &kimi_k3_decode::shared_experts::validate_residency, "",
          pybind11::arg("active_tokens"),
          pybind11::arg("available_sms"));
    m.def("_kimi_k3_shared_experts_generation_advanced",
          &kimi_k3_decode::shared_experts::generation_advanced, "",
          pybind11::arg("observed"),
          pybind11::arg("consumed"));
    m.def("_kimi_k3_shared_experts_wait_timeout_clocks", []() {
        return kimi_k3_decode::shared_experts::kGenerationWaitTimeoutClocks;
    });
    m.def("_kimi_k3_shared_experts_wait_timed_out",
          &kimi_k3_decode::shared_experts::wait_timed_out, "",
          pybind11::arg("started"),
          pybind11::arg("current"));
    m.def("_kimi_k3_shared_experts_timeout_metadata",
          &kimi_k3_decode::shared_experts::timeout_metadata_for_testing);
    m.def("_kimi_k3_tail", &kimi_k3_decode::kimi_k3_tail_entrypoint, "",
          pybind11::arg("routed_latent_rmsnorm_weight"),
          pybind11::arg("latent_up_proj"),
          pybind11::arg("collective_buffer"),
          pybind11::arg("collective_buffer_ptrs"),
          pybind11::arg("collective_buffer_multicast_ptr"),
          pybind11::arg("output_mailbox"),
          pybind11::arg("output_mailbox_ptrs"),
          pybind11::arg("output_mailbox_multicast_ptr"),
          pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"),
          pybind11::arg("scratch"),
          pybind11::arg("error_flag"),
          pybind11::arg("tp_rank"),
          pybind11::arg("active_tokens"),
          pybind11::arg("workspace_signature"));
    m.def("_kimi_k3_workspace_signature",
          &kimi_k3_decode::kimi_k3_workspace_signature_entrypoint, "",
          pybind11::arg("collective_buffer"),
          pybind11::arg("collective_buffer_ptrs"),
          pybind11::arg("collective_buffer_multicast_ptr"),
          pybind11::arg("output_mailbox"),
          pybind11::arg("output_mailbox_ptrs"),
          pybind11::arg("output_mailbox_multicast_ptr"),
          pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("tp_rank"));
    m.def("_kimi_k3_tail_shared_memory_reservations",
          &kimi_k3_decode::tail::shared_memory_reservations_for_testing, "",
          pybind11::arg("device"));
    m.def("_kimi_k3_tail_role_plan",
          &kimi_k3_decode::tail::role_plan_for_testing, "",
          pybind11::arg("active_tokens"));
    m.def("_kimi_k3_tail_validate_residency",
          &kimi_k3_decode::tail::validate_residency, "",
          pybind11::arg("active_tokens"),
          pybind11::arg("available_sms"));
    m.def("_kimi_k3_tail_generation_advanced",
          &kimi_k3_decode::tail::generation_advanced, "",
          pybind11::arg("observed"), pybind11::arg("consumed"));
    m.def("_kimi_k3_tail_barrier_reached",
          &kimi_k3_decode::tail::barrier_reached, "",
          pybind11::arg("observed"), pybind11::arg("target"));
    m.def("_kimi_k3_tail_wait_timeout_clocks", []() {
        return kimi_k3_decode::tail::kGenerationWaitTimeoutClocks;
    });
    m.def("_kimi_k3_tail_wait_timed_out",
          &kimi_k3_decode::tail::wait_timed_out, "",
          pybind11::arg("started"), pybind11::arg("current"));
    m.def("_kimi_k3_tail_timeout_metadata",
          &kimi_k3_decode::tail::timeout_metadata_for_testing);
    m.def("pack_kimi_k3_mxfp4", &kimi_k3_decode::mxfp4::pack_entrypoint, "",
          pybind11::arg("weight"), pybind11::arg("padded_k"));
    m.def("dequant_kimi_k3_mxfp4", &kimi_k3_decode::mxfp4::dequant_entrypoint, "",
          pybind11::arg("packed"), pybind11::arg("scale"),
          pybind11::arg("logical_k"));
    m.def("_kimi_k3_mixed_mma_probe",
          &kimi_k3_decode::expert_mxfp4::mixed_mma_probe_entrypoint, "",
          pybind11::arg("a"), pybind11::arg("b_packed"),
          pybind11::arg("b_scale"));
    m.def("all_gather_top_experts", &utils::all_gather_top_experts::all_gather_top_experts_entrypoint, "",
          pybind11::arg("top_experts"), pybind11::arg("all_gather_top_experts_buffer"),
          pybind11::arg("all_gather_top_experts_buffer_multicast_ptr"), pybind11::arg("rank"), pybind11::arg("chunk_bytes"));
    m.def("barrier_all", &utils::barrier_all::barrier_all_entrypoint, "",
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"), pybind11::arg("target"));
    m.def("_barrier_all_wait_timeout_clocks",
          &utils::barrier_all::barrier_all_wait_timeout_clocks, "");
    m.def("schedule", &scheduler::schedule_entrypoint, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("schedule_capacity"), pybind11::arg("rank"));
    m.def("mxfp8_quantize", &mxfp8::quantize_entrypoint, "",
          pybind11::arg("x_bf16"),
          pybind11::arg("return_normal"), pybind11::arg("return_transposed"));
    m.def("dispatch_mlp_swiglu_combine_fwd_mxfp8", &dispatch_mlp_swiglu_combine_fwd_mxfp8_entrypoint, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"), pybind11::arg("w_routed_down_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_bwd_mxfp8", &dispatch_mlp_swiglu_combine_bwd_mxfp8_entrypoint, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate_T"), pybind11::arg("w_routed_gate_T_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up_T"), pybind11::arg("w_routed_up_T_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down_T"), pybind11::arg("w_routed_down_T_sc"),
          pybind11::arg("x_fp8_t_routed"), pybind11::arg("x_sc_t_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_fp8_routed"), pybind11::arg("gate_sc_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_fp8_routed"), pybind11::arg("up_sc_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_fp8_t_routed"), pybind11::arg("hidden_sc_t_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_fwd_bf16", &dispatch_mlp_swiglu_combine_fwd_bf16_entrypoint, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_bwd_bf16", &dispatch_mlp_swiglu_combine_bwd_bf16_entrypoint, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("x_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("fwd_epilogue", &utils::fwd_epilogue::fwd_epilogue_entrypoint, "",
          pybind11::arg("y_shared"), pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"));
    m.def("bwd_epilogue", &utils::bwd_epilogue::bwd_epilogue_entrypoint, "",
          pybind11::arg("d_x_shared"), pybind11::arg("d_x_routed_buffer"));
    m.def("recompute_forward_context_mxfp8", &recompute_forward_context_mxfp8_entrypoint, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("recompute_forward_context_bf16", &recompute_forward_context_bf16_entrypoint, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
}
