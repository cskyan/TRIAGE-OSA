# Uploading this prepared repository

Target repository: <https://github.com/cskyan/TRIAGE-OSA>

The prepared directory contains source code, documentation, example inputs
and two ZIP files under `source_data/`. It is not itself a Git clone.

1. Clone the existing GitHub repository into a separate working directory
   using GitHub Desktop or `git clone https://github.com/cskyan/TRIAGE-OSA.git`.
2. Copy the contents of this prepared directory into that clone, including
   `.gitignore` and `.gitattributes`. Keep the clone's `.git` directory.
3. Review the changes. Both named source-data ZIP files should appear in the
   changes list. They are explicitly allowed by `.gitignore`.
4. Commit the reviewed files, then push using GitHub Desktop or Git. Do not
   force-push or replace existing history.
5. On GitHub, verify the README, both archive download links, and the example
   directory. Download the two archives and compare their hashes with
   `source_data/SHA256SUMS.txt`.

The first archive is approximately 31.1 MiB, exceeding GitHub's 25 MiB
per-file browser-upload limit. Use Git or GitHub Desktop for this package.
Both archives are below GitHub's 100 MiB regular-Git per-file limit; LFS is
not necessary for their current sizes. See the
[GitHub size-limit documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Do not add raw patient records, local result trees, private credentials,
environment folders or unapproved model weights. The packaged test and
example outputs are generated under ignored `tmp/` and `outputs/` folders.
