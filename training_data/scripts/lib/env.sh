#!/usr/bin/env bash

pdcash_trim() {
  local value="$1"
  value="${value#"${value%%[!$' \t\r\n']*}"}"
  value="${value%"${value##*[!$' \t\r\n']}"}"
  printf '%s' "${value}"
}

pdcash_strip_dotenv_quotes() {
  local value="$1"
  local first
  local last
  if [[ ${#value} -ge 2 ]]; then
    first="${value:0:1}"
    last="${value: -1}"
    if [[ "${first}" == "${last}" && ( "${first}" == "'" || "${first}" == '"' ) ]]; then
      printf '%s' "${value:1:${#value}-2}"
      return
    fi
  fi
  printf '%s' "${value}"
}

pdcash_load_dotenv() {
  local dotenv_path="${1:-}"
  [[ -n "${dotenv_path}" && -f "${dotenv_path}" ]] || return 0

  local raw_line
  local line
  local key
  local value
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="$(pdcash_trim "${raw_line}")"
    [[ -n "${line}" ]] || continue
    [[ "${line:0:1}" != "#" ]] || continue
    [[ "${line}" == *"="* ]] || continue

    if [[ "${line}" == export[[:space:]]* ]]; then
      line="$(pdcash_trim "${line#export}")"
    fi

    key="$(pdcash_trim "${line%%=*}")"
    value="$(pdcash_trim "${line#*=}")"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -z "${!key+x}" ]] || continue

    value="$(pdcash_strip_dotenv_quotes "${value}")"
    export "${key}=${value}"
  done < "${dotenv_path}"
}

pdcash_resolve_log_path() {
  local value="$1"
  local project_root="${PROJECT_ROOT:?PROJECT_ROOT must be set}"
  local log_root="${LOG_ROOT:-${project_root}/logs/slurm}"
  case "${value}" in
    /*)
      printf '%s' "${value}"
      ;;
    logs/slurm)
      printf '%s' "${log_root}"
      ;;
    logs/slurm/*)
      printf '%s/%s' "${log_root%/}" "${value#logs/slurm/}"
      ;;
    *)
      printf '%s/%s' "${project_root%/}" "${value}"
      ;;
  esac
}

pdcash_resolve_data_path() {
  local value="$1"
  local project_root="${PROJECT_ROOT:?PROJECT_ROOT must be set}"
  local data_root="${DATA_ROOT:-${project_root}/data}"
  case "${value}" in
    /*)
      printf '%s' "${value}"
      ;;
    data)
      printf '%s' "${data_root}"
      ;;
    data/*)
      printf '%s/%s' "${data_root%/}" "${value#data/}"
      ;;
    *)
      printf '%s/%s' "${project_root%/}" "${value}"
      ;;
  esac
}
