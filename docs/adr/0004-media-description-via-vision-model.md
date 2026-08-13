# Media handling for text-only upstreams uses a vision model to describe images

When a local request carries images and the upstream is a confirmed text-only model, the proxy replaces the image blocks with a text description produced by a separately configured vision model (`VISION_URL` / `VISION_API_KEY` / `VISION_MODEL`, format detected from the URL). All images in one request are analyzed together in a single non-streaming vision call; results are cached in-process keyed by image fingerprint with a TTL.

This mirrors how Plugin-Deepseek-Vision makes DeepSeek "understand" images via a host vision model, but with our own config surface instead of a plugin host. The sanitizer's previous behavior (replace with a `[Unsupported Image]` placeholder) was deliberately weaker and is kept only as the no-vision-config fallback.
