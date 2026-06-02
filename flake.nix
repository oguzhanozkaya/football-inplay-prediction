{
  description = "turkish-inflation-forecasting flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      project = "turkish-inflation-forecasting";

      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      base = {
        packages = with pkgs; [
          just
        ];

        env = { };

        shellHook = "";
      };

      shells = {
        dev = {
          packages = with pkgs; [
            nixfmt
            nixd
            mermaid-cli
            cudatoolkit
            bun
            uv
            pkg-config
          ];

          env = {
            CUDA_PATH = "${pkgs.cudatoolkit}";
          };
        };

        docs = {
          packages = with pkgs; [
            uv
          ];
        };

      };
    in
    {
      devShells.${system} =
        builtins.mapAttrs
          (
            shellName: shell:
            let
              name = "${project}_${shellName}_shell";
              packages = pkgs.lib.unique (base.packages ++ (shell.packages or [ ]));
              shellHook = pkgs.lib.concatStringsSep "\n" [
                base.shellHook
                (shell.shellHook or "")
                ''echo "> ${name}"''
              ];
              env = (base.env or { }) // (shell.env or { });
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
            in
            pkgs.mkShell {
              inherit
                name
                packages
                env
                shellHook
                LD_LIBRARY_PATH
                ;
            }
          )
          (
            shells
            // {
              default = {
                packages = pkgs.lib.unique (
                  pkgs.lib.flatten (map (shell: shell.packages or [ ]) (builtins.attrValues shells))
                );
                env = pkgs.lib.foldl' (acc: env: acc // env) { } (
                  map (shell: shell.env or { }) (builtins.attrValues shells)
                );
                shellHook = pkgs.lib.concatStringsSep "\n" (
                  map (shell: shell.shellHook or "") (builtins.attrValues shells)
                );
              };
            }
          );
    };
}
