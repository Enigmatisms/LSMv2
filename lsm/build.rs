extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-lcuda")
        .flag("-lcudart")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .file("cuda/ray_tracer.cu")
        .file("cuda/host_func.cu")
        .compile("libcuda_helper.a");
}

// "nvcc" "-ccbin=c++" "-Xcompiler" "-O0" "-Xcompiler" "-ffunction-sections" "-Xcompiler" "-fdata-sections" "-Xcompiler" "-fPIC" "-G" "-Xcompiler" "-g" "-Xcompiler" "-fno-omit-frame-pointer" "-m64" "-Xcompiler" "-Wall" "-Xcompiler" "-Wextra" "-cudart=shared" "-gencode" "aaarch=compute_86,code=sm_86" "-o" "/home/sentinel/various/Rusty/c_test/target/debug/build/c_test-ee663f0370f4cb55/out/src/cuda_test.o" "-c" "--device-c" "src/cuda_test.cu"
