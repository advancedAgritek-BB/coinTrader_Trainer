from __future__ import annotations

"""Helper for uploading artifacts to Supabase Storage."""

import logging
import os
from typing import Tuple

from supabase import Client, create_client

from config import load_config


def _parse_dest(dest: str, src_path: str) -> Tuple[str, str]:
    """Return the bucket name and object path for ``dest``.

    Parameters
    ----------
    dest : str
        Destination in the format ``"bucket/path/in/bucket"``.  If only the
        bucket is provided, the basename of ``src_path`` is used as the object
        name.
    src_path : str
        Local path to the file being uploaded.  Used when ``dest`` only
        specifies the bucket.
    """
    if "/" in dest:
        bucket, object_name = dest.split("/", 1)
    else:
        bucket, object_name = dest, os.path.basename(src_path)
    return bucket, object_name


def upload_to_supabase(path: str, dest: str) -> None:
    """Upload a local file to Supabase Storage.

    Parameters
    ----------
    path : str
        Path to the local file to upload.
    dest : str
        Destination in the format ``"bucket/path/in/bucket"`` or simply the
        bucket name. When only the bucket name is supplied, the basename of
        ``path`` is used as the object name.
    """
    cfg = load_config()
    url = cfg.supabase_url
    key = cfg.supabase_service_key or cfg.supabase_key
    client: Client = create_client(url, key)

    bucket, object_name = _parse_dest(dest, path)
    logging.debug("Uploading %s to bucket %s as %s", path, bucket, object_name)
    with open(path, "rb") as fh:
        client.storage.from_(bucket).upload(object_name, fh)
