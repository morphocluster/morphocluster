echo "Regenerating conda-lock.yml..."

conda-lock -f environment.base.yml -f environment.dev.yml --lockfile conda-lock.yml -p linux-64 --without-cuda