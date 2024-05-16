# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.make
    pkgs.python3
    pkgs.python3Packages.pip 
    pkgs.python3Packages.pytorch
  ];

  shellHook = ''
    echo "Welcome to your C and Python development environment"
  '';
}

