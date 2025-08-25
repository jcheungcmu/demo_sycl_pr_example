//==============================================================
// Iota is the equivalent of a Hello, World! sample for data parallel programs.
// Building and running the sample verifies that your development environment
// is setup correctly and demonstrates the use of the core features of SYCL.
// This sample runs on both CPU and GPU (or FPGA). When run, it computes on both
// the CPU and offload device, then compares results. If the code executes on
// both CPU and the offload device, the name of the offload device and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::seconds, milliseconds, etc.

#include <dlfcn.h>
// #include "/home/jcheung2/ofs24/oneapi-asp/common/source/include/aocl_mmd.h"
// #include "/home/jcheung2/ofs24/oneapi-asp/common/source/include/mmd.h"

// #if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
  #include <sycl/ext/intel/fpga_extensions.hpp>
// #endif

using namespace sycl;
using namespace std;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};


//************************************
// Demonstrate iota both sequential on CPU and parallel on device.
//************************************
int main() {
//   // Create device selector for the device of your interest.
// #if FPGA_EMULATOR
//   // Intel extension: FPGA emulator selector on systems without FPGA card.
//   auto selector = sycl::ext::intel::fpga_emulator_selector_v;
// #elif FPGA_SIMULATOR
//   // Intel extension: FPGA simulator selector on systems without FPGA card.
//   auto selector = sycl::ext::intel::fpga_simulator_selector_v;
// #elif FPGA_HARDWARE
//   // Intel extension: FPGA selector on systems with FPGA card.
//   auto selector = sycl::ext::intel::fpga_selector_v;
// #else
//   // The default device selector will select the most performant device.
//   auto selector = default_selector_v;
// #endif


  cout << "Starting...\n";
  auto platforms = sycl::platform::get_platforms();

  cout << "Getting platforms\n";

  for (auto platform : sycl::platform::get_platforms())
  {
      std::cout << "\n\n\n\nPlatform: "
                << platform.get_info<sycl::info::platform::name>()
                << std::endl;

      for (auto device : platform.get_devices())
      {
          std::cout << "\n\n\n\n\t****************Device: "
                    << device.get_info<sycl::info::device::name>()
                    << std::endl;
      }
  }

  // dynamic loading flow for splitting kernels across homogeneous fpgas

  auto slot0_lib = dlopen("/home/jcheung2/multi_fpga/pr_test/slot0/pr_src.so", RTLD_NOW);
  auto slot0_src     = (event (*)(queue&, buffer<long>&, size_t, long, int))dlsym(slot0_lib, "pr_src");
  auto slot0_pr_request     = (event (*)(queue&, buffer<int>&))dlsym(slot0_lib, "pr_request_kernel");
  auto slot0_pr_ack     = (void (*)(queue&, int))dlsym(slot0_lib, "pr_ack_kernel");

  auto slot1_lib1 = dlopen("/home/jcheung2/multi_fpga/pr_test/slot1/pr_sink1.so", RTLD_NOW);
  auto slot1_sink1     = (event (*)(queue&, buffer<long>&, buffer<size_t>&, size_t))dlsym(slot1_lib1, "pr_sink1");
  auto slot1_stop1     = (void (*)(queue&))dlsym(slot1_lib1, "stop_req_kernel1");

  auto slot1_lib2 = dlopen("/home/jcheung2/multi_fpga/pr_test/slot1/pr_sink2.so", RTLD_NOW);
  auto slot1_sink2     = (event (*)(queue&, buffer<long>&, buffer<size_t>&, size_t))dlsym(slot1_lib2, "pr_sink2");
  auto slot1_stop2     = (void (*)(queue&))dlsym(slot1_lib2, "stop_req_kernel2");

  // size_t N = 1000000000; // 8 GB long
  size_t N = 10;

  std::vector<long> src_mem(N);
  std::vector<long> sink_mem(N);

  int pr_request[1] = {0};
  size_t work_status[1] = {0};

  // initial current module
  int current_module = 1;

  long offset = 0;
  long threshold = 0;
  for (size_t i = 0; i < N; i++) {
    if (i%2 == 0) {
      src_mem[i] = 1; // Fill with some data
    }
    else {
      src_mem[i] = -1; // Fill with some data
    }
    sink_mem[i] = 0; // Initialize sink memory
  }

  try {

    buffer<long> buf_sink_mem(&sink_mem[0], N);
    buffer<long> buf_src_mem(&src_mem[0], N);

    buffer<int> buf_pr_request(&pr_request[0], 1);
    buffer<size_t> buf_work_status(&work_status[0], 1);

    // slots appear as two different fpga devices
    cout << "CREATING Q0\n";
    queue q0(platforms[1].get_devices()[0], exception_handler);
    queue q0_pr_ctrl(platforms[1].get_devices()[0], exception_handler);
    
    cout << "CREATING Q1\n";
    queue q1(platforms[1].get_devices()[1], exception_handler);
    queue q1_pr_ctrl(platforms[1].get_devices()[1], exception_handler);

    // Print out the device information used for the kernel code.
    cout << "Src running on device: "
         << q0.get_device().get_info<info::device::name>() << "\n";

    // Print out the device information used for the kernel code.
    cout << "Sink running on device: "
         << q1.get_device().get_info<info::device::name>() << "\n";

    // by default initialize slot1 with sink1

    cout << "submitting sink1 (default) to slot1\n";
    auto ev_sink = slot1_sink1(q1, buf_sink_mem, buf_work_status, 0);

    cout << "submitting src to slot0 with N = " << N << "\n";
    auto ev_src = slot0_src(q0, buf_src_mem, N, threshold,  current_module);
    auto ev_pr_req = slot0_pr_request(q0_pr_ctrl, buf_pr_request);


    while (ev_src.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {

      cout << "Waiting for PR request...\n";
      ev_pr_req.wait();
      host_accessor<int> host_pr_request(buf_pr_request);

      int request_id = host_pr_request[0];
      if (request_id != 0) {
        cout << "PR request received.\n";
        cout << "Request ID: " << request_id << "\n";

        if (current_module == 1) {
          cout << "Sending stop request to sink1 kernel.\n";
          slot1_stop1(q1_pr_ctrl); // Stop the sink1 kernel
        }
        else if (current_module == 2) { 
          cout << "Sending stop request to sink2 kernel.\n";
          slot1_stop2(q1_pr_ctrl); // Stop the sink2 kernel
        }

        cout << "Waiting for sink to finish...\n";
        ev_sink.wait(); // Wait for sink1 to finish
        cout << "Slot1 ready to be reconfigured.\n";

        { // need accessor to go out of scope to allow reconfiguration
          host_accessor<size_t> host_work_status(buf_work_status);
          size_t resume_addr = host_work_status[0];

          if (request_id == 1) {
            cout << "Reconfiguring Slot1 to Sink1 and set to resume at address: " << resume_addr << "\n";
            ev_sink = slot1_sink1(q1, buf_sink_mem, buf_work_status, resume_addr);
          }
          else if (request_id == 2) {
            cout << "Reconfiguring Slot1 to Sink2 and set to resume at address: " << resume_addr << "\n";
            ev_sink = slot1_sink2(q1, buf_sink_mem, buf_work_status, resume_addr);
          }
        }

        while(ev_sink.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::running) {
          cout << "Waiting for PR to complete...\n";
          std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for a bit before checking again
        }

        cout << "PR completed for request ID: " << request_id << "\n";

        // update the current module
        current_module = request_id;

        // Reset PR request monitor
        ev_pr_req = slot0_pr_request(q0_pr_ctrl, buf_pr_request);
        cout << "PR request monitor reset.\n";
        // Send PR ack
        slot0_pr_ack(q0_pr_ctrl,  current_module);
        cout << "PR acknowledgment sent for request ID: " << request_id << "\n";
      }
    }

    cout << "**************FINISHED*************\n";
    cout << "Src kernel completed. Stopping sink\n";
    if (current_module == 1) {
      cout << "Sending stop request to sink1 kernel.\n";
      slot1_stop1(q1_pr_ctrl); // Stop the sink1 kernel
    }
    else if (current_module == 2) { 
      cout << "Sending stop request to sink2 kernel.\n";
      slot1_stop2(q1_pr_ctrl); // Stop the sink2 kernel
    }


    host_accessor<long> result_sink_mem(buf_sink_mem);

    cout << "*************CHECKING*************\n";

    int fail = 0;
    for (size_t i = 0; i < N; i++) {

      long expected_value;
      if (src_mem[i] < threshold) {
        expected_value = src_mem[i] + 1; // Sink1 logic
      } else {
        expected_value = src_mem[i] + 3; // Sink2 logic
      }
      if (result_sink_mem[i] != expected_value) {
        fail++;
      }

      if (i < 10 || i == N - 1) { // Print first 10 and last element
        cout << "sink_mem[" << i << "] = " << result_sink_mem[i] << "\n";
      }
    }

    if (fail > 0) {
      cout << "Sink memory test failed with " << fail << " errors.\n";
    } else {
      cout << "Sink memory test passed successfully.\n";
    }

  } catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
  }

  cout << "Successfully completed on device1.\n";
  cout << "Successfully completed on device2.\n";
  return 0;


}
