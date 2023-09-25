{ pkgs ? import (fetchTarball
  # around NixOS 23.05
  "https://github.com/NixOS/nixpkgs/archive/9034b46dc4c7596a87ab837bb8a07ef2d887e8c7.tar.gz")
  { } }:
(pkgs.buildFHSUserEnv rec {
  name = "sdfCAD-env";
  targetPkgs = pkgs:
    (with pkgs; [
      python3
      poetry
      zlib # for numpy
    ]);
  profile = ''
    # create and fill virtual environment
    poetry install --all-extras --with=dev
    # enter virtual environment
    source "$(poetry env info --path)"/bin/activate
    # Don't store notebook outputs
    nbstripout --install
    if ! python -c 'import tkinter' 2>/dev/null >/dev/null;then
      # Fix tkinter import (shouldn't be necessary, but it is... Including tkinter in targetPkgs doesn't help)
      export PYTHONPATH="${
        toString pkgs.python3Packages.tkinter
      }/lib/python3.10/site-packages/:$PYTHONPATH"
    fi
    export LD_LIBRARY_PATH="${
      with pkgs; lib.makeLibraryPath [ xorg.libX11 xorg.libXi xorg.libXrender libGL ]
    }:$LD_LIBRARY_PATH"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
  '';
}).env
