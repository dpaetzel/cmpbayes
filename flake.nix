{
  inputs = {
    nixpkgs.url =
      "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";
    overlays.url = "github:dpaetzel/overlays";
  };

  outputs = { self, nixpkgs, overlays }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # That overlay contains arviz.
        overlays = [ overlays.pymc4 ];
      };
      python = pkgs.python39;
    in {
      # TODO In order to provide a proper default flake here we need to package
      # pystan/httpstan properly.

      devShell.${system} = pkgs.mkShell {

        # We use ugly venvShellHook here because packaging pystan/httpstan is
        # not entirely straightforward.
        buildInputs = with pkgs;
          with python3Packages; [
            python
            venvShellHook
            numpy
            pymc4
            pandas
          ];

        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }:$LD_LIBRARY_PATH";
        '';

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install pystan==3.4.0
        '';

      };
    };
}
