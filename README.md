# GaussianCutSim

工具の解析的な掃引と有限Gaussian Splat（GS）を使い、切削後の表面を計算・表示するWindows向けツールです。CUDA Tool-profile Dexelで計算し、PySide6の操作画面とOpen3D-2Cut2DGSで結果を表示します。

正式版のバージョンは `0.1.0` です。

## 必要な環境

- Windows x64
- Python 3.12 x64
- NVIDIA製のCUDA対応GPUと、CUDA 12系を利用できるNVIDIAドライバー
- Git
- 初回セットアップ用のインターネット接続

Open3Dをソースからビルドする場合は、次の環境も必要です。

- Visual Studioの「C++によるデスクトップ開発」（MSVC x64/x86とWindows SDK）
- CMake 3.24以上、かつ使用するVisual Studioに対応したバージョン

セットアップはバッチファイルとPythonで動作し、PowerShellは不要です。CUDA計算は必須で、CPUへの自動切替はありません。

**描画には拡張版Open3D-2Cut2DGSが必要です。標準版Open3Dは使用せず、セットアップでもインストールしません。**

### セットアップ前に用意するもの

`setup.bat` はPython・Git・Visual Studio・NVIDIAドライバー自体をインストールしません。Open3Dをソースからビルドする場合は、先に次を準備してください。

| 用意するもの | 導入・確認内容 | 入手先 |
| --- | --- | --- |
| Python 3.12 x64 | Python 3.12の64ビット版を用意します。`py -3.12` で起動できる状態、または `GS_SETUP_PYTHON` で実行ファイルを指定できる状態にします。 | [Python公式](https://www.python.org/downloads/windows/) |
| Git for Windows | コマンドプロンプトで `git` が使えるようにします。サブモジュール取得にも必要です。 | [Git公式](https://git-scm.com/install/windows) |
| Visual Studio／Build Tools | 「C++によるデスクトップ開発」と、MSVC x64/x86・Windows SDKを導入します。 | [MicrosoftのC++環境導入手順](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) |
| CMake | 3.24以上で、使用するVisual Studioのジェネレーターに対応したものを用意します。Visual Studio付属のCMakeも使用できます。 | [CMake公式](https://cmake.org/download/) |
| NVIDIAドライバー | 使用するGPUに対応し、CUDA 12系の実行に対応したドライバーを導入します。 | [NVIDIA公式](https://www.nvidia.com/en-us/drivers/) |

### Visual Studio Installerで選ぶ項目

1. Visual Studio Installerを開き、使用するVisual StudioまたはBuild Toolsの「変更」を選びます。
2. ワークロードで「C++によるデスクトップ開発」を選びます。
3. インストール内容に **MSVCのx64/x86ビルドツール** と **Windows SDK** が含まれていることを確認します。
4. Visual Studio付属のCMakeを使う場合は、**Windows用C++ CMakeツール**も選びます。
5. インストールを完了し、コマンドプロンプトを開き直します。

Visual Studio Codeだけでは、必要なC++ビルド環境は揃いません。現在のビルドスクリプトはVisual Studio 2019・2022・2026の検出に対応しています。CMakeはバージョン番号の下限を満たすだけでなく、選んだVisual Studioに対応している必要があります。

### 準備できたか確認する

コマンドプロンプトで次を実行してください。

```bat
py -3.12 -c "import sys, struct; print(sys.version); print(struct.calcsize('P') * 8, 'bit')"
git --version
nvidia-smi
```

Pythonが `3.12.x`・`64 bit` と表示され、GitのバージョンとNVIDIA GPU・ドライバーの情報が表示されることを確認します。`py` を使わない場合は、Python実行ファイルのフルパスで同じ確認ができます。

CMakeを単独でインストールしてPATHに追加した場合は、次も確認します。

```bat
cmake --version
```

Visual Studio付属のCMakeはPATHに入っていなくてもスクリプトが検出します。ただし、PATHに別のCMakeがある場合はそちらが優先されます。明示的に指定するには `setup.bat --cmake-exe "C:\path\to\cmake.exe"` を使用してください。

準備ができたら、次節のクローンと `setup.bat` に進みます。`.venv` 作成後は `build_open3d.bat --check-only` でビルド前提条件を確認できます。このコマンドは初回の `.venv` 作成前には使えません。

通常のセットアップではCuPy用CUDAコンポーネントも導入するため、CUDA Toolkitの手動インストールは不要です。Open3D自体はCUDAモジュールを無効にしてビルドし、切削計算はCuPyでGPUを使用します。NVIDIAドライバーは別途必要です。

ソース・ビルド中間ファイル・wheel・仮想環境を保存する空き容量と、依存ファイルを取得できるネット接続も確保してください。メモリ不足でビルドが失敗する場合は、`setup.bat --jobs 1` で並列数を減らせます。

## クローンから起動まで

以下はコマンドプロンプト（`cmd.exe`）で実行します。

```bat
git clone --recurse-submodules https://github.com/yangyf-works/GaussianCutSim.git
cd GaussianCutSim
setup.bat
.venv\Scripts\python.exe run_app.py
```

すでにクローン済みで `Open3D/` の内容がない場合は、先に次を実行してください。

```bat
git submodule update --init --recursive
```

`setup.bat` は次の処理を行います。

1. リポジトリ直下に `.venv` を作成します。既存の有効な `.venv` は再利用します。
2. `requirements.txt` の依存パッケージと、CuPy用のCUDAコンポーネントを導入します。
3. 導入済みの2Cut2DGSが利用できれば保持します。未導入、または標準版Open3Dしかない場合は、`Open3D/` の固定コミットをビルドして導入します。
4. パッケージの整合性、2Cut2DGSの拡張API、CUDAカーネルのコンパイル・GPU実行、標準加工データの読み込みを確認します。

`Environment validation passed.` が表示されれば環境確認は完了です。初回のOpen3Dビルドには時間がかかります。

通常の起動では `.venv` の有効化操作は不要です。リポジトリ直下で次を実行します。

```bat
.venv\Scripts\python.exe run_app.py
```

### ビルド済み2Cut2DGS wheelを使う場合

Python 3.12・Windows x64向けの拡張版wheelがある場合は、ソースビルドを省略できます。

```bat
setup.bat --open3d-wheel "C:\packages\open3d-拡張版-cp312-cp312-win_amd64.whl"
```

パスは実際のwheelファイルに置き換えてください。必要な2Cut2DGS APIを持たないwheelは拒否します。この方法ではOpen3DをビルドするためのVisual Studio・CMakeは不要です。

### 環境の確認とオプション

```bat
rem インストールせずに、現在の環境とGPU実行を確認
setup.bat --check-only

rem インストール済みのCUDA 12 Toolkitを使用
setup.bat --use-system-cuda

rem オプション一覧
setup.bat --help
```

通常はオプションなしで実行してください。`--use-system-cuda` は、CuPy用CUDAコンポーネントの追加導入を省略する指定です。NVIDIAドライバーはどちらの方法でも別途必要です。

Pythonを自動検出できない場合は、実行ファイルを指定できます。

```bat
set "GS_SETUP_PYTHON=C:\path\to\Python312\python.exe"
setup.bat
```

## Open3Dの再ビルド

`Open3D/` は、`yangyf-works/Open3D-2Cut2DGS` の `codex/open3d-2cut2dgs` ブランチを参照するサブモジュールです。実際のビルドにはGaussianCutSim側で記録したコミットを使用し、ブランチの最新コミットへ自動更新しません。

有効な `.venv` がある状態で実行します。インストール時にファイルを使用しないよう、GaussianCutSimは終了してください。

```bat
rem ビルド前提条件のみ確認
build_open3d.bat --check-only

rem ビルドして .venv にインストール
build_open3d.bat
```

出力先は既定で `build/open3d/` です。並列ビルド数は `--jobs 4` のように指定できます。

指定コミットとの不一致、Git管理対象ファイルの変更、未追跡のソース・設定ファイルがある場合は、ビルドを止めます。ただし、`Open3D/docs/` 内の未追跡の `.rst`・`.md` 文書は許可します。文書の追加・変更によって自動で再ビルドする仕組みではありません。

## 加工サンプル

引数なしの起動では、Flat/Ballの12工程・36切削ブロックを読み込みます。代表サンプルは次の4種類です。

| フォルダ | 内容 | 切削ブロック数 |
| --- | --- | ---: |
| `machining_data/mixed_demo` | Flat/Ball 12工程。突込み・加工・引上げの基本例 | 36 |
| `machining_data/cavity_demo` | 浅い曲面金型5工程。荒取りと底・壁の仕上げ | 394 |
| `machining_data/indexed_demo` | 割り出し5軸5姿勢。上面・側面・45度の固定姿勢加工 | 15 |
| `machining_data/mold` | 金型5工程。外形・上面・荒加工・自由曲面仕上げ・細部加工 | 1775 |

```bat
.venv\Scripts\python.exe run_app.py --project machining_data/cavity_demo
.venv\Scripts\python.exe run_app.py --project machining_data/indexed_demo
.venv\Scripts\python.exe run_app.py --project machining_data/mold
```

`--project` にはデータフォルダまたは `project.json` を指定します。`--data-dir` で画面に表示する加工データの検索先を変更できます。

## 基本操作

1. 加工データを開き、`Compute All` で計算します。
2. 計算済みの結果を再生し、ブロック選択やシークで加工状態を確認します。
3. 視点、工具・経路の表示、工具別の色、GSの表示サイズ、材質を調整します。

GSを `Ctrl`＋左ダブルクリックすると、そのGSに対応する加工の工具・経路を表示します。

`GS density` は計算に使うGS配置密度の設定です。`GS size` は表示サイズの設定です。起動時に密度を指定する場合は、例えば `--gs-density 0.8` で80%に設定できます（指定範囲は `0.10`～`8.0`、既定値は `1.0`）。

Open3D単独の操作画面を使う場合は、次を実行します。

```bat
.venv\Scripts\python.exe run_app.py --open3d-ui
```

## 加工データの作成

`examples/` に、加工データ作成用のプロンプトとJSON例があります。

- `examples/create_machining_data_prompt.txt`：AIに加工条件とともに渡す作成用プロンプト。読み込み確認のコマンドも記載しています。
- `examples/machining_example/`：素材・工具・経路を分けたJSON例。Flat工具とBall工具の2工程・6切削ブロックです。
- `examples/three_axis_program.json`：単独の指令配列を使う簡易形式の例です。

標準の加工データは、同じフォルダに次の3ファイルを置きます。

| ファイル | 内容 |
| --- | --- |
| `project.json` | データ名、素材範囲、単位、工具・経路ファイルの参照 |
| `tools.json` | 工具ID、Flat／Ball種別、半径などの工具条件 |
| `toolpath.json` | 工具の開始・終了位置、使用工具、移動種別など |

```bat
.venv\Scripts\python.exe run_app.py --project examples/machining_example
.venv\Scripts\python.exe run_app.py --program examples/three_axis_program.json
```

`--program` の簡易形式は既定の素材を使用します。素材も指定したい場合は `--project` の3ファイル形式を使用してください。

## 対応範囲

- Flat／Ball工具によるXYZ直線移動
- 固定姿勢の割り出し5軸（3＋2軸）加工
- 突込み・引上げ・ランプ加工・異なる工具による再切削
- 計算済み結果の再生、工具・経路表示、GSの密度・表示サイズ・材質設定

同時5軸、円弧補間、NCファイルの直接読み込み、プロジェクト保存、実行時の衝突検出は未対応です。再生は計算済みのGS差分を使い、途中位置で切削を再計算するものではありません。

表面は有限GSとClippingによる近似です。微小形状や、任意の視点での完全な被覆を保証するものではありません。

## フォルダ構成

```text
GaussianCutSim/
├─ run_app.py              アプリの起動入口
├─ setup.bat               .venv作成・依存導入・実行環境の確認
├─ build_open3d.bat        2Cut2DGSのビルド・インストール
├─ requirements.txt        Python依存パッケージ
├─ .gitmodules             Open3Dサブモジュールの参照設定
├─ cutsim/                 計算・描画・操作画面・実行時診断
├─ machining_data/         代表加工サンプル4種類
├─ examples/               加工データのJSON例・作成用プロンプト
├─ tools/                  セットアップ・ビルドなどの補助処理
├─ Open3D/                 Open3D-2Cut2DGSサブモジュール
├─ .venv/                  セットアップ時に作成するPython環境
└─ build/                  Open3Dのビルド出力
```

`cutsim/cuda`・`cutsim/diagnostics` は現在のアプリからも使用するため、実行用ソースに含めています。

このリポジトリには正式版アプリ、環境構築スクリプト、加工サンプルと入力例を含めています。仮想環境・ビルド出力・ログ・キャッシュはGit管理対象外です。

## ログ・初回計算について

`[PERF] Log: ...` は性能ログの保存先を示す通常のメッセージです。初回計算ではCUDAカーネルなどのコンパイルが入り、時間がかかる場合があります。CUDAカーネルの既定キャッシュ先は `%LOCALAPPDATA%\GaussianCutSim\cuda-cache` です。

CuPyから `CUDA path could not be detected` という警告が出ても、それだけで計算失敗を意味するわけではありません。次の確認で `CUDA kernel compilation and execution: OK` と表示されるかを確認してください。

```bat
setup.bat --check-only
```

実際にCUDA確認が失敗する場合は、表示されたエラーとNVIDIAドライバーを確認してください。`--use-system-cuda` を指定していた場合はCUDA 12 Toolkitの設定を確認するか、オプションなしでセットアップを再実行してください。
