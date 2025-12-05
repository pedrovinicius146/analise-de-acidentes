
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIGURAÇÕES DE DIRETÓRIOS
# ============================

BASE_DIR = "./BASES"
OUT_CSV = "./RESULTADOS_CSV"
OUT_GRAFICOS = "./RESULTADOS_GRAFICOS"
OUT_LIMPOS = "./RESULTADOS_LIMPOS"

# Criar diretórios de saída caso não existam
for p in [OUT_CSV, OUT_GRAFICOS, OUT_LIMPOS]:
    os.makedirs(p, exist_ok=True)

# ============================
# LOCALIZAR ARQUIVOS CSV
# ============================

pattern = os.path.join(BASE_DIR, "*acidentes*.csv")
files = glob.glob(pattern)

if len(files) == 0:
    raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão {pattern}")

print("Arquivos encontrados:")
for f in files:
    print(" -", f)

# ============================
# FUNÇÃO ROBUSTA PARA LEITURA
# ============================

def try_read_csv(path):
    tried = []
    for sep in [';', ',', ';']:
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                print(f"Leitura OK: sep='{sep}', enc='{enc}' -> shape {df.shape}")
                return df
            except Exception as e:
                tried.append((sep, enc, str(e)))

    print("Falhou, primeiras tentativas com erro:", tried[:5])
    raise ValueError("Falha ao ler CSV")


# ============================
# CARREGAR E UNIFICAR BASES
# ============================

dfs = []
for file in files:
    df_tmp = try_read_csv(file)
    df_tmp["__source_file"] = os.path.basename(file)
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True, sort=False)
print("Dimensão após concat:", df.shape)

# ============================
# PADRONIZAÇÃO DE COLUNAS
# ============================

df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Detectar coluna de data automaticamente
date_candidates = [c for c in df.columns if "data" in c]
date_col = "data_inversa" if "data_inversa" in df.columns else (date_candidates[0] if date_candidates else None)

# Converter datas
if date_col:
    df["event_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
else:
    df["event_date"] = pd.NaT

df["year"] = df["event_date"].dt.year
df["month_str"] = df["event_date"].dt.strftime("%Y-%m")

# UF / Estado
if "uf" in df.columns:
    df["state"] = df["uf"].astype(str).str.strip().str.upper()

# Cidade
city_col = next((c for c in ["municipio", "cidade"] if c in df.columns), None)
df["city"] = df[city_col].astype(str).str.title() if city_col else None

# Causa
cause_col = next((c for c in df.columns if "causa" in c), None)
df["cause"] = df[cause_col].astype(str).str.lower() if cause_col else None

# Converter colunas numéricas de vítimas
sev_cols = [c for c in df.columns if any(x in c for x in ["morto", "ferid", "vitim"])]
for c in sev_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================
# EXPORTAR CSVs BÁSICOS
# ============================

years_df = df["year"].value_counts().sort_index().reset_index()
years_df.columns = ["year", "count"]
years_df.to_csv(f"{OUT_CSV}/contagem_por_ano_acidentes.csv", index=False)

ts_month = df.groupby("month_str").size().reset_index(name="accidents_count")
ts_month.to_csv(f"{OUT_CSV}/acidentes_por_mes.csv", index=False)

top_states = df["state"].value_counts().reset_index()
top_states.columns = ["state", "count"]
top_states.to_csv(f"{OUT_CSV}/top_estados_acidentes.csv", index=False)

top_causes = df["cause"].value_counts().reset_index()
top_causes.columns = ["cause", "count"]
top_causes.to_csv(f"{OUT_CSV}/top_causas_acidentes.csv", index=False)

cols_save = ["event_date", "year", "month_str", "state", "city", "cause"] + sev_cols
df[cols_save].to_csv(f"{OUT_LIMPOS}/acidentes_limpo_unificado.csv", index=False)

# ============================
# GRÁFICOS BÁSICOS
# ============================

plt.figure(figsize=(12,5))
plt.plot(ts_month["month_str"], ts_month["accidents_count"])
plt.xticks(rotation=60)
plt.title("Acidentes por mês")
plt.tight_layout()
plt.savefig(f"{OUT_GRAFICOS}/plot_acidentes_por_mes.png")
plt.close()

top15 = top_states.head(15).iloc[::-1]
plt.figure(figsize=(10,6))
plt.barh(top15["state"], top15["count"])
plt.title("Top 15 estados por número de acidentes")
plt.tight_layout()
plt.savefig(f"{OUT_GRAFICOS}/top15_estados_acidentes.png")
plt.close()

# ============================
# ANÁLISES AVANÇADAS
# ============================

print("\n=== Gerando análises avançadas... ===\n")

# --- Acidentes por dia da semana ---
weekday_count = df['event_date'].dt.day_name().value_counts().reset_index()
weekday_count.columns = ["weekday", "count"]
weekday_count.to_csv(f"{OUT_CSV}/acidentes_por_dia_semana.csv", index=False)

plt.figure(figsize=(10,5))
plt.bar(weekday_count['weekday'], weekday_count['count'])
plt.title("Acidentes por dia da semana")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_GRAFICOS}/acidentes_por_dia_semana.png")
plt.close()


# --- Heatmap Ano x Mês ---
pivot = df.pivot_table(index="year", columns=df["event_date"].dt.month, aggfunc="size", fill_value=0)

plt.figure(figsize=(12,6))
plt.imshow(pivot, aspect="auto")
plt.colorbar(label="Número de acidentes")
plt.xticks(range(12), [f"{m:02d}" for m in range(1,13)])
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Heatmap - Acidentes por Ano x Mês")
plt.savefig(f"{OUT_GRAFICOS}/heatmap_ano_mes.png")
plt.close()

pivot.to_csv(f"{OUT_CSV}/heatmap_ano_mes.csv")

# --- Top Cidades ---
top_cities = df["city"].value_counts().head(20).reset_index()
top_cities.columns = ["city", "count"]
top_cities.to_csv(f"{OUT_CSV}/top20_cidades.csv", index=False)

plt.figure(figsize=(10,8))
plt.barh(top_cities["city"], top_cities["count"])
plt.title("Top 20 cidades com mais acidentes")
plt.tight_layout()
plt.savefig(f"{OUT_GRAFICOS}/top20_cidades.png")
plt.close()

# --- Gravidade por Estado ---
grav_cols = [c for c in df.columns if "morto" in c or "ferid" in c]

if grav_cols:
    gravidade = df.groupby("state")[grav_cols].mean().reset_index()
    gravidade.to_csv(f"{OUT_CSV}/gravidade_por_estado.csv", index=False)

    plt.figure(figsize=(12,6))
    plt.bar(gravidade["state"], gravidade[grav_cols[0]])
    plt.title(f"Média de {grav_cols[0]} por estado")
    plt.tight_layout()
    plt.savefig(f"{OUT_GRAFICOS}/gravidade_por_estado.png")
    plt.close()

# --- Correlação ---
if len(grav_cols) >= 2:
    corr_df = df[grav_cols].corr()
    corr_df.to_csv(f"{OUT_CSV}/correlacao_vitimas.csv")

    plt.figure(figsize=(6,5))
    plt.imshow(corr_df, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(grav_cols)), grav_cols, rotation=45)
    plt.yticks(range(len(grav_cols)), grav_cols)
    plt.title("Correlação entre variáveis de vítimas")
    plt.tight_layout()
    plt.savefig(f"{OUT_GRAFICOS}/correlacao_vitimas.png")
    plt.close()

print("=== Análises avançadas concluídas! ===\n")

# ============================
# FINALIZAÇÃO
# ============================

print("\n===============================")
print("Processo concluído!")
print("CSV →", OUT_CSV)
print("Gráficos →", OUT_GRAFICOS)
print("Base limpa →", OUT_LIMPOS)
print("===============================\n")
