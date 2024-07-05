{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;

      config = {
        allowUnfree = true;
      };
    };
  in {
    devShells.${system}.default =
      pkgs.mkShell
      {
        name = "cuda-env-shell";

        nativeBuildInputs = with pkgs; [
          autoconf
          binutils
          cudaPackages.cudatoolkit
          curl
          freeglut
          gcc12
          git
          gitRepo
          gnumake
          gnupg
          gperf
          libGL
          libGLU
          linuxPackages.nvidia_x11
          m4
          ncurses5
          procps
          stdenv.cc
          unzip
          util-linux
          xorg.libX11
          xorg.libXext
          xorg.libXi
          xorg.libXmu
          xorg.libXrandr
          xorg.libXv
          zlib
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
        '';
      };
  };
}
