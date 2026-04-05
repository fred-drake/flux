{
  description = "Flux image generation dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python3
            python3Packages.pip
            python3Packages.virtualenv
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              python3 -m venv .venv
            fi
            source .venv/bin/activate
            if [ ! -f .venv/.installed ]; then
              echo "Installing dependencies..."
              pip install -r requirements.txt
              touch .venv/.installed
            fi
          '';
        };
      }
    );
}
