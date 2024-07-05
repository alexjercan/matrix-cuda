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
          git
          gcc12
          gitRepo
          gnupg
          autoconf
          curl
          procps
          gnumake
          util-linux
          m4
          gperf
          unzip
          cudaPackages.cudatoolkit
          linuxPackages.nvidia_x11
          libGLU
          libGL
          xorg.libXi
          xorg.libXmu
          freeglut
          xorg.libXext
          xorg.libX11
          xorg.libXv
          xorg.libXrandr
          zlib
          ncurses5
          stdenv.cc
          binutils
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
