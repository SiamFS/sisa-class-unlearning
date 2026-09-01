"""Project-directory cloning (W34).

A "project" is one directory under config.PROJECTS_DIR holding sisa_data/, models/,
data_info/ and logs/. Several workflows need to work on a THROWAWAY copy rather than
mutate the real one: the W8 scratch reference, the exactness eval's unlearned system,
and unlearning test mode (config.UNLEARNING_TEST_MODE).

Lives in utils/ rather than experiments/scratch_reference.py because
unlearning/sisa_unlearning.py needs it too, and scratch_reference imports
SISAUnlearning -- putting it there would be a circular import.

The .npy files are hard-linked rather than copied: sisa_data/ is ~95% of a project's
bytes and a clone's data is identical to the source's apart from the shard that gets
stripped. See clone_file for why that is safe.
"""
import os
import shutil

__all__ = ['clone_file', 'clone_project']


def clone_file(src: str, dst: str) -> None:
    """copytree's per-file action: hard-link .npy files, real-copy everything else.

    Copying sisa_data byte-for-byte cost ~700 MB of disk writes per clone (and
    exactness_eval does two), against ~19 MB for the models/plots/JSON that actually
    diverge. A hard link costs a directory entry.

    Safe only because of the replace-don't-truncate invariant in
    SISAUnlearning._save_slice_data: it np.saves to a temp path and os.replace()s it,
    which installs a NEW inode and leaves the source's link intact. That is the only
    writer that touches a slice .npy inside a clone (entry_data_processing.py always
    writes config.PROJECT_NAME, never a clone). A writer that instead opened one of
    these paths 'wb' would truncate straight through the link into the pristine source,
    so keep that invariant if you add one.

    Falls back to a real copy if the link can't be made -- different volume, or a
    filesystem without hard links.
    """
    if src.endswith('.npy'):
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def clone_project(source_project: str, dest_project: str, projects_dir: str = None) -> str:
    """Clone `source_project` to `dest_project`, returning the destination path.

    Replaces `dest_project` if it already exists. `projects_dir` defaults to
    config.PROJECTS_DIR (imported lazily so this module stays importable from
    anywhere in the tree without import-order surprises).
    """
    if projects_dir is None:
        import config
        projects_dir = config.PROJECTS_DIR

    src = os.path.join(projects_dir, source_project)
    dst = os.path.join(projects_dir, dest_project)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source project not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=clone_file)
    return dst
