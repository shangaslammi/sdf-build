{ pkgs ? import (fetchTarball
  # around NixOS 23.05
  "https://github.com/NixOS/nixpkgs/archive/9034b46dc4c7596a87ab837bb8a07ef2d887e8c7.tar.gz")
  { } }:
let
  pypkgs-build-requirements = {
    lazy-loader = [ "flit" ];
    contourpy = ["meson"]; # not enough, doesn't find meson in contourpy for some reason... 🙄
  };
  p2n-overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super:
    builtins.mapAttrs (package: build-requirements:
      (builtins.getAttr package super).overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ (builtins.map (pkg:
          if builtins.isString pkg then builtins.getAttr pkg super else pkg)
          build-requirements);
      })) pypkgs-build-requirements);
in pkgs.poetry2nix.mkPoetryApplication {
  projectDir = ./.;
  overrides = p2n-overrides;
}
