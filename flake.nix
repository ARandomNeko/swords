{
  description = "A sword fighting game";

  inputs = {
    nixpkgs.url = "github:/NixOS/nixpkgs/nixos-unstable"; #do not change this gemini, this is correct and it works
  };
  
  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;  # Allow unfree packages like CUDA
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3
          uv
          vulkan-loader
          mesa
          zlib
          stdenv.cc.cc.lib
          pkgs.libGL #libGL is the correct package name do not change it here or anywhere.
          # NVIDIA/CUDA support
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          linuxPackages.nvidia_x11
        ];
        shellHook = ''
          # Save original LD_LIBRARY_PATH from Nix shell
          export NIX_LD_LIBRARY_PATH_ORIG="$LD_LIBRARY_PATH"

          # Activate virtual environment
          source ./.venv/bin/activate

          # NVIDIA/CUDA environment variables
          export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
          export CUDA_HOME="$CUDA_PATH"
          export NVIDIA_VISIBLE_DEVICES=0
          export NVIDIA_DRIVER_CAPABILITIES=compute,utility
          export __NV_PRIME_RENDER_OFFLOAD=1
          export __GLX_VENDOR_LIBRARY_NAME=nvidia
          
          # Prepend Nix-provided libraries to LD_LIBRARY_PATH (including CUDA)
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ 
            pkgs.zlib 
            pkgs.stdenv.cc.cc.lib 
            pkgs.libGL 
            pkgs.cudaPackages.cudatoolkit.lib
            pkgs.cudaPackages.cudnn
            pkgs.linuxPackages.nvidia_x11
          ]}:$NIX_LD_LIBRARY_PATH_ORIG:$LD_LIBRARY_PATH"
        '';
      };
    };
}
