"""Load a document into a business's knowledge store.

    python ingest.py knowledge/clinic_policies.md --profile profiles/clinic.json
    python ingest.py --profile profiles/clinic.json --list
    python ingest.py --profile profiles/clinic.json --forget clinic_policies.md
    python ingest.py knowledge/clinic_policies.md --profile profiles/clinic.json --dry-run

The business is named by its profile rather than typed in, so a document can
only ever be filed under a business that actually exists — a mistyped tenant id
would otherwise create a silent orphan store that nothing ever reads.

This is the command-line form of what the upload button will eventually do. The
UI will call the same functions in knowledge.py.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import knowledge  # noqa: E402
from prompt import business_id, load_profile  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Load a document into a business's knowledge store.")
    parser.add_argument("document", nargs="?", help="path to a .md or .txt file")
    parser.add_argument("--profile", default="profiles/clinic.json",
                        help="which business this belongs to (default: profiles/clinic.json)")
    parser.add_argument("--list", action="store_true", help="show what is already stored")
    parser.add_argument("--forget", metavar="SOURCE", help="remove one document by its filename")
    parser.add_argument("--dry-run", action="store_true", help="show the chunks, store nothing")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    bid = business_id(profile)
    name = profile.get("business", {}).get("name", bid)
    print(f"\n  {name}  (id: {bid})\n")

    knowledge.init_knowledge()

    if args.list:
        stored = knowledge.sources(bid)
        if not stored:
            print("  Nothing stored yet.\n")
            return
        print(f"  {'Document':34}{'Chunks':>8}   Ingested")
        print(f"  {'-' * 62}")
        for s in stored:
            print(f"  {s['source'][:33]:34}{s['chunks']:>8}   {s['ingested']}")
        print()
        return

    if args.forget:
        removed = knowledge.forget(bid, args.forget)
        print(f"  Removed {removed} chunks.\n" if removed
              else f"  Nothing stored under {args.forget!r}.\n")
        return

    if not args.document:
        parser.error("give a document to ingest, or use --list / --forget")

    path = Path(args.document)
    if not path.exists():
        sys.exit(f"  No such file: {path}")

    chunks = knowledge.chunk(path.read_text(encoding="utf-8"))
    if not chunks:
        sys.exit("  That file produced no chunks — is it empty?")

    print(f"  {path.name} -> {len(chunks)} chunks\n")
    for i, c in enumerate(chunks):
        first = c.splitlines()[0][:64]
        print(f"    {i:>2}. {first:<66} {len(c):>5} chars")
    print()

    if args.dry_run:
        print("  DRY RUN — nothing stored.\n")
        return

    # Embedding costs money and writing replaces what is there, so the last
    # word is a human's.
    if input("  Store these? [y/N] ").strip().lower() != "y":
        sys.exit("  Nothing stored.")

    vectors = knowledge.embed(chunks)
    count = knowledge.store(bid, path.name, chunks, vectors, business_name=name)
    print(f"\n  Stored {count} chunks under {bid}.\n")


if __name__ == "__main__":
    main()
