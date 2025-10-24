{
  description = "Rust example flake for Zero to Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    devenv.url = "github:cachix/devenv";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    {
      nixpkgs,
      devenv,
      systems,
      rust-overlay,
      ...
    }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (
        system:
        let
          overlays = [ (import rust-overlay) ];
          pkgs = nixpkgs.legacyPackages.${system}.extend (
            final: prev: {
              rustPkgs = import nixpkgs {
                inherit system overlays;
              };
            }
          );
          rust-toolchain = pkgs.rustPkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                packages = with pkgs; [
                  rust-toolchain
                  stdenv.cc.cc.lib
                  mold
                  clang
                  pkg-config
                  cargo-binutils

                  libGL
                  alsa-lib
                  libudev-zero
                  vulkan-loader
                  wayland
                  libxkbcommon

                  cargo-watch
                  cargo-flamegraph
                ];

                # Set up environment variables for Bevy/Wayland
                env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                  pkgs.libxkbcommon
                  pkgs.libGL
                  pkgs.vulkan-loader
                  pkgs.wayland
                  pkgs.alsa-lib
                  pkgs.libudev-zero
                ];
              }
            ];
          };
        }
      );
    };
}
