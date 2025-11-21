from wsindy_manifold.latent.anim import ensure_writer


def test_ensure_writer_returns_string():
    writer = ensure_writer()
    assert isinstance(writer, str)
    assert writer in {"ffmpeg", "pillow"}
