{
  description = "A Nix-flake-based C/C++ development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
  inputs.nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";


  outputs = { self, nixpkgs, nixpkgs-unstable }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
        unstable = import nixpkgs-unstable { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs, unstable }:
        let
          mypkgs = with pkgs; [
            stdenv
            rdma-core.dev
            clang-tools
            pkg-config
            cmake
            ninja
            gnumake
            gcc11
            cudaPackages.cuda_cudart
            cudaPackages.cudatoolkit
            cudaPackages.libcublas
            # Create nvshmem package later on
            libunwind
            zstd
            openmpi
            python311
            linuxKernel.packages.linux_6_1.nvidia_x11
          ] ++
          (with pkgs.python311Packages; [ pybind11 numpy ]);
        in
        {
          default = pkgs.mkShell {
            buildInputs = mypkgs;
            nativeBuildInputs = mypkgs;
            LD_LIBRARY_PATH = "${pkgs.lib.strings.makeLibraryPath (mypkgs)}";
          };
        });
    };
}
