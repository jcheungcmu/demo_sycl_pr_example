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

// #if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
// #endif

using namespace sycl;
using namespace std;

// // Create an exception handler for asynchronous SYCL exceptions
// static auto exception_handler = [](sycl::exception_list e_list) {
//   for (std::exception_ptr const &e : e_list) {
//     try {
//       std::rethrow_exception(e);
//     }
//     catch (std::exception const &e) {
// #if _DEBUG
//       std::cout << "Failure" << std::endl;
// #endif
//       std::terminate();
//     }
//   }
// };

// // Array type and data size for this example.
// constexpr size_t array_size = 10000;
// typedef array<int, array_size> IntArray;

//************************************
// Iota in SYCL on device.
//************************************

template <unsigned ID>
struct ethernet_pipe_id {
  static constexpr unsigned id = ID;
};

using read_iopipe = ext::intel::kernel_readable_io_pipe<ethernet_pipe_id<0>, long, 0>;

using write_iopipe = ext::intel::kernel_writeable_io_pipe<ethernet_pipe_id<0>, long, 0>;

using pr_request_pipe = ext::intel::pipe<class PipePRRequest, int, 0>;
using pr_ack_pipe = ext::intel::pipe<class PipePRack, int, 0>;


extern "C" {
  event pr_src(queue &q, buffer<long>& a_buf, size_t num_items, long threshold, int initial_module) {

    return q.submit([&](handler &h) {
      // Create an accessor with read permission.
      accessor a(a_buf, h, read_only);

      h.single_task<class iopipes_src_test> ([=]() { 

        long readdata = 0;
        long writedata = 0; 
        int current_module = initial_module;

        size_t i = 0;

        for (i = 0; i < num_items; i++) {

          readdata = a[i];

          if (readdata < threshold) {
            if (current_module != 1) {
              pr_request_pipe::write(1);
              current_module = pr_ack_pipe::read(); // Wait for acknowledgment, current module should be 1
            }

            if (current_module == 1) 
              writedata = readdata + 1;

          } else {
            if (current_module != 2) {
              pr_request_pipe::write(2);
              current_module = pr_ack_pipe::read(); // Wait for acknowledgment, current module should be 2
            }

            if (current_module == 2) 
              writedata = readdata + 2;
          }

          write_iopipe::write(writedata);


        }

        // close pr_request_kernel 
        if (i >= num_items) {
          pr_request_pipe::write(0); // Write 0 to indicate end of requests
        }
      });

    });
  }
  
  event pr_request_kernel(queue &q, buffer<int>& a_buf) {

    return q.submit([&](handler &h) {
      accessor a(a_buf, h, write_only, no_init);

      h.single_task<class pr_request> ([=]() { 

        int read_request;
        read_request = pr_request_pipe::read();

        a[0] = read_request; // Store the initial request in the first element of the buffer
      });

    });
  }

  // void pr_request_kernel(queue &q, int& request_id) {

  //   q.submit([&](handler &h) {
  //     // Create an accessor with read permission.
  //     // accessor a(a_buf, h, write_only, no_init);

  //     // int read_request;

  //     h.single_task<class pr_request> ([=]() { 

  //       request_id = pr_request_pipe::read();

  //     });

  //   });
  // }

  void pr_ack_kernel(queue &q, int module_id) {

    q.submit([&](handler &h) {

      h.single_task<class pr_ack> ([=]() { 

        pr_ack_pipe::write(module_id);
      });

    });
  }
  
}

// //************************************
// // Demonstrate iota both sequential on CPU and parallel on device.
// //************************************
// int main() {
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

//   // Create array objects with "array_size" to store data.
//   IntArray sequential, parallel;
//   constexpr int value = 2;

//   // Sequential iota.
//   for (size_t i = 0; i < sequential.size(); i++) sequential[i] = value + i;

//   try {
//     queue q(selector, exception_handler);

//     // Print out the device information used for the kernel code.
//     cout << "Running on device: "
//          << q.get_device().get_info<info::device::name>() << "\n";
//     cout << "Array size: " << parallel.size() << "\n";

//     // Parallel iota in SYCL.
//     IotaAddParallel(q, parallel, value);
//   } catch (std::exception const &e) {
//     cout << "An exception is caught while computing on device.\n";
//     terminate();
//   }

//   // Verify two results are equal.
//   for (size_t i = 0; i < sequential.size(); i++) {
//     if (parallel[i] != sequential[i]) {
//       cout << "Failed on device.\n";
//       return -1;
//     }
//   }

//   int indices[]{0, 1, 2, (sequential.size() - 1)};
//   constexpr size_t indices_size = sizeof(indices) / sizeof(int);

//   // Print out iota result.
//   for (int i = 0; i < indices_size; i++) {
//     int j = indices[i];
//     if (i == indices_size - 1) cout << "...\n";
//     cout << "[" << j << "]: " << j << " + " << value << " = "
//          << parallel[j] << "\n";
//   }

//   cout << "Successfully completed on device.\n";
//   return 0;
// }
