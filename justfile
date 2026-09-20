fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q

# the release the dogfood serves; bump deliberately, this is the deployment pin
SACCADE_RELEASE ?= "v0.1.0"

# restart the dogfood server on the pinned release; the deployment lags HEAD by design
serve:
    @nix run "git+file://$PWD?ref=$(SACCADE_RELEASE)" -- serve \
        --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811 \
        </dev/null >>/tmp/sac-serve.log 2>&1 &
    @sleep 2
    @curl -sf -o /dev/null http://127.0.0.1:8811/ && echo "serving :8811 on $(SACCADE_RELEASE) (log: /tmp/sac-serve.log)" || (echo "NOT up — /tmp/sac-serve.log tail:"; tail -3 /tmp/sac-serve.log; exit 1)

# the workspace build, for smokes and development only
serve-dev:
    cargo build
    @pkill -x sac || pkill -x saccade || true
    @sleep 1
    @nohup setsid env -u SACCADE_DB ./target/debug/sac serve --bind 0.0.0.0 --port 8811 </dev/null >>/tmp/sac-serve.log 2>&1 &
    @sleep 1
    @curl -sf -o /dev/null http://127.0.0.1:8811/ && echo "serving :8811 (dev build)" || (echo "NOT up — /tmp/sac-serve.log tail:"; tail -3 /tmp/sac-serve.log; exit 1)
