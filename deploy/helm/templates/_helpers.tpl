{{/* Shared helpers for the freqtrade-ai chart. */}}
{{- define "freqtrade-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "freqtrade-ai.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "freqtrade-ai.labels" -}}
helm.sh/chart: {{ include "freqtrade-ai.chart" . }}
app.kubernetes.io/name: {{ include "freqtrade-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "freqtrade-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "freqtrade-ai.freqtradeTag" -}}
{{- /* ftctl renders the image variant as .Values.freqtrade.tag; fall back to stable. */ -}}
{{- coalesce .Values.freqtrade.image.tag .Values.freqtrade.tag "stable" -}}
{{- end -}}
