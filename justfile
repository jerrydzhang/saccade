# The repository's recipes: one justfile per project, loaded as modules
# (saccade/justfile as saccade::, app/justfile as app::). Each project's
# recipes run from its own directory and enter its own shell, so the
# root needs no shell of its own. fmt, lint and test here run both
# projects; the project recipes the repository has always named keep
# their names at the root as aliases.

mod saccade
mod app

fmt: saccade::fmt app::fmt
    direnv exec saccade nixfmt flake.nix nix/*.nix saccade/flake-module.nix saccade/nix/*.nix app/flake-module.nix app/nix/*.nix

lint: saccade::lint app::lint

# the full suite: saccade's (the cargo workspace, the native node
# suites), then the app's (the app shell's own tests, the app
# frontend's vitest unit and browser projects; the vitest native
# project is gated: just app-test-native)
test: saccade::test app::test

alias typecheck := saccade::typecheck
alias native-test := saccade::native-test
alias shell-test := app::shell-test
alias client-deps := app::client-deps
alias app-test := app::app-test
alias app-test-browser := app::app-test-browser
alias app-test-pending := app::app-test-pending
alias app-test-native := app::app-test-native
alias viewer-check := app::viewer-check
alias port-debug-bin := app::port-debug-bin
alias viewer-shell := app::viewer-shell
