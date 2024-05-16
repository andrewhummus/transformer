{
  description = "A flake for an enhanced C and Python development environment with PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = { self, nixpkgs }: {
    devShells = {
      default = nixpkgs.lib.mkShell {
        buildInputs = [
          nixpkgs.gcc
          nixpkgs.make
          nixpkgs.python310Full
          nixpkgs.python310Packages.pytorch
          nixpkgs.cmake
          nixpkgs.gdb
          nixpkgs.git
          nixpkgs.python310Packages.virtualenv
        ];

        # Environment variables
        PYTHONPATH = "${nixpkgs.python310Packages.pytorch}/${nixpkgs.python310Packages.numpy}";

        # Shell hooks
        preShellHook = ''
          echo "Setting up the environment..."
          export PATH=$PATH:${nixpkgs.python310}/bin
        '';

        shellHook = ''
          echo "Welcome to your enhanced C and Python development environment"
          source ${nixpkgs.python310Packages.virtualenv}/bin/virtualenvwrapper.sh
          if [ ! -d "$VIRTUAL_ENV" ]; then
            mkvirtualenv transformer_env
            workon transformer_env
            pip install torch 
          fi
        '';
      };
    };

    packages = {
      default = nixpkgs.mkDerivation {
        pname = "transformer";
        version = "0.1.0";
        src = ./.;

        buildInputs = [
          nixpkgs.gcc
          nixpkgs.make
          nixpkgs.python310Full
          nixpkgs.python310Packages.pip 
          nixpkgs.python310Packages.pytorch
          nixpkgs.cmake
          nixpkgs.gdb
          nixpkgs.git
        ];

        # Custom build phases if needed
        buildPhase = ''
          echo "Building the project..."
          make
        '';

        installPhase = ''
          echo "Installing the project..."
          mkdir -p $out/bin
          cp main $out/bin
        '';
      };
    };
  };
}

