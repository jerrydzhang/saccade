fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q

# rebuild, then (re)start the dogfood server on :8811; verifies it answers
serve:
    cargo build
    @pkill -x sac || true
    @sleep 1
    @nohup setsid env -u SACCADE_DB ./target/debug/sac serve --bind 0.0.0.0 --port 8811 </dev/null >>/tmp/sac-serve.log 2>&1 &
    @sleep 1
    @curl -sf -o /dev/null http://127.0.0.1:8811/ && echo "serving :8811 (log: /tmp/sac-serve.log)" || (echo "NOT up — /tmp/sac-serve.log tail:"; tail -3 /tmp/sac-serve.log; exit 1)
