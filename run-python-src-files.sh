find src/ -name *.py | grep -v  __init__ | sed -e 's/\//./g' | rev | cut -d '.' -f 2- | rev | xargs -I '{}' -t sh -c 'python3 -m {} >/dev/null || exit 255'
