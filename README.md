# GaussianCutSim

GaussianCutSimは、工具の移動に伴う材料の除去と加工面の変化を3Dで確認できる、Windows向けの切削シミュレーションツールです。Flat／Ball工具による3軸加工と、工具姿勢を固定した割り出し5軸（3＋2軸）加工に対応しています。

工具の解析的な掃引とDexelによる切削計算をCUDAで実行し、加工面を有限Gaussian Splat（GS）とClippingで表現します。描画には独自拡張版のOpen3D-2Cut2DGS、操作画面にはPySide6を使用しています。

- 加工ブロックごとの計算・再生と、任意の視点からの形状確認
- 工具別の加工面の色分け、工具・経路の表示
- 基本加工・金型・割り出し加工のサンプルと、加工データ作成用のJSON例

正式版：**0.1.0**

## 1. 必要な環境

Windows x64、NVIDIA製のCUDA対応GPU、インターネット接続と、ビルド用の空き容量を用意してください。次のソフトウェアは、あらかじめ導入します。

| ソフトウェア | 要件 |
| --- | --- |
| [Python](https://www.python.org/downloads/windows/) | 3.12 x64。`py -3.12` で起動できること |
| [Git for Windows](https://git-scm.com/install/windows) | コマンドプロンプトで `git` が使えること |
| [NVIDIAドライバー](https://www.nvidia.com/en-us/drivers/) | 使用するGPUとCUDA 12系の実行に対応すること |
| [Visual Studio／Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) | 「C++によるデスクトップ開発」、MSVC x64/x86、Windows SDK |
| [CMake](https://cmake.org/download/) | 3.24以上、かつ使用するVisual Studioに対応したバージョン |
| [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) | Windows x64版。`glslangValidator.exe` を含むこと |

**ビルド済みのOpen3D-2Cut2DGS wheelを使用する場合、Visual Studio・CMake・Vulkan SDKは不要です。** 標準版Open3Dは使用できません。

### NVIDIA GPUドライバーの更新

古いドライバーが必要なCUDAの実行に対応していない場合は、更新が必要です。

1. [NVIDIA公式サイト](https://www.nvidia.com/en-us/drivers/)から、使用するGPUとWindowsに対応するドライバーを入手・インストールします。
2. 再起動を求められた場合はPCを再起動し、`nvidia-smi` でGPU名とドライバーバージョンを確認します。
3. 次節の `setup.bat` で、CUDAカーネルのコンパイルとGPU実行が成功することを確認します。

`setup.bat` はドライバーを更新しません。ビルド済みwheelを使う場合も、対応ドライバーは必要です。

### Open3Dをビルドするための準備

Visual Studio Installerで「C++によるデスクトップ開発」を選び、**MSVC x64/x86・Windows SDK・Windows用C++ CMakeツール**を導入してください。ビルドスクリプトはVisual Studio 2019・2022・2026の検出に対応しています。Visual Studio Codeだけではビルドできません。

Gaussian Splatting描画用のシェーダーコンパイラを用意するため、**Vulkan SDKのインストールも必須**です。

1. [Vulkan SDK公式サイト](https://vulkan.lunarg.com/sdk/home#windows)からWindows x64版をダウンロードします。
2. 標準設定でインストールします。コンパイラの格納フォルダが環境変数 `Path` に自動追加されます。
3. コマンドプロンプトを開き直し、`glslangValidator --version` で動作を確認します。

## 2. クローン・セットアップ・起動

以下をコマンドプロンプト（`cmd.exe`）で実行します。PowerShellは不要です。

```bat
git clone --recurse-submodules https://github.com/yangyf-works/GaussianCutSim.git
cd GaussianCutSim
setup.bat
.venv\Scripts\python.exe run_app.py
```

`setup.bat` は `.venv` の作成、Python依存パッケージとCuPy用CUDAコンポーネントの導入、必要に応じたOpen3D-2Cut2DGSのビルド、GPU実行の確認を行います。**`Environment validation passed.` が表示されれば環境確認は完了です。** 初回のビルドには時間がかかります。

Python・Git・ビルドツール・GPUドライバー自体は自動導入しません。通常はCUDA Toolkitの手動インストールは不要です。CUDA計算は必須で、CPUへの自動切替はありません。

次回からは `.venv\Scripts\python.exe run_app.py` だけで起動できます。仮想環境の有効化操作は不要です。

### ビルド済みwheelを使う場合

Python 3.12・Windows x64向けの**Open3D-2Cut2DGS wheel**がある場合は、上の `setup.bat` を次に置き換えます。パスは実際のファイルに変更してください。

```bat
setup.bat --open3d-wheel "C:\packages\open3d-拡張版-cp312-cp312-win_amd64.whl"
```

<details>
<summary>セットアップの補足・Open3Dの再ビルド</summary>

- クローン済みで `Open3D/` が空の場合：`git submodule update --init --recursive`
- 環境・GPU実行の確認：`setup.bat --check-only`
- ビルド時のメモリ不足：`setup.bat --jobs 1` で並列数を減らす
- Pythonの指定：`set "GS_SETUP_PYTHON=C:\path\to\Python312\python.exe"` の後に `setup.bat`
- CMakeの指定：`setup.bat --cmake-exe "C:\path\to\cmake.exe"`（通常はPATHまたはVisual Studioから検出）
- 導入済みCUDA 12 Toolkitを使用：`setup.bat --use-system-cuda`
- オプション一覧：`setup.bat --help`

Open3Dの再ビルドは、有効な `.venv` がある状態でアプリを終了してから実行します。

```bat
build_open3d.bat --check-only
build_open3d.bat
```

ビルド出力は `build/open3d/`、インストール先は `.venv` です。並列数は `--jobs 4` などで指定できます。

`Open3D/` は [Open3D-2Cut2DGS](https://github.com/yangyf-works/Open3D-2Cut2DGS) の `codex/open3d-2cut2dgs` ブランチを参照するサブモジュールです。ビルドには本リポジトリが記録したコミットを使い、最新版へ自動更新しません。コミットの不一致、管理対象ファイルの変更、未追跡のソース・設定がある場合は停止します。`Open3D/docs/` 内の未追跡 `.rst`・`.md` 文書は許可され、文書の変更だけでは自動再ビルドしません。

</details>

## 3. 基本操作と加工サンプル

1. 加工データを開き、`Compute All` で計算します。
2. 計算済みの結果を再生し、ブロック選択やシークで加工状態を確認します。
3. 視点、工具・経路の表示、工具別の色、GSの表示サイズ、材質を調整します。

GSを **`Ctrl`＋左ダブルクリック**すると、そのGSに対応する加工の工具・経路を表示します。`GS density` は計算に使うGS配置密度、`GS size` は表示サイズです。

引数なしではFlat/Ballの標準デモを読み込みます。

| データフォルダ（`machining_data/` 内） | 内容 | 切削ブロック数 |
| --- | --- | ---: |
| `mixed_demo` | Flat/Ball 12工程。突込み・加工・引上げ | 36 |
| `cavity_demo` | 浅い曲面金型5工程。荒取り・底と壁の仕上げ | 394 |
| `indexed_demo` | 割り出し5軸5姿勢。上面・側面・45度の固定姿勢加工 | 15 |
| `mold` | 金型5工程。外形・上面・荒加工・自由曲面仕上げ・細部加工 | 1775 |

別のサンプルを開く例：

```bat
.venv\Scripts\python.exe run_app.py --project machining_data/mold
```

`--project` にはデータフォルダまたは `project.json` を指定します。その他の起動オプションは `.venv\Scripts\python.exe run_app.py --help` で確認できます。

## 4. 加工データの作成

[作成用プロンプト](examples/create_machining_data_prompt.txt)と[JSON例](examples/machining_example/)を用意しています。標準形式は、同じフォルダに次の3ファイルを置きます。

| ファイル | 内容 |
| --- | --- |
| `project.json` | データ名、素材範囲、単位、工具・経路ファイルの参照 |
| `tools.json` | 工具ID、Flat／Ball種別、半径など |
| `toolpath.json` | 工具の開始・終了位置、使用工具、移動種別など |

```bat
.venv\Scripts\python.exe run_app.py --project examples/machining_example
```

[簡易形式の例](examples/three_axis_program.json)は `--program examples/three_axis_program.json` で読み込めます。この形式では既定の素材を使うため、素材を指定する場合は上記の3ファイル形式を使用してください。

## 対応範囲と制限

Flat／Ball工具のXYZ直線移動、固定姿勢の割り出し加工、突込み・引上げ・ランプ加工、異なる工具による再切削に対応します。同時5軸、円弧補間、NCファイルの直接読み込み、プロジェクト保存、実行時の衝突検出は未対応です。

再生には計算済みのGS差分を使い、途中位置で切削を再計算しません。表面は有限GSとClippingによる近似で、微小形状や任意の視点での完全な被覆を保証するものではありません。

## 起動・計算で困ったとき

- **初回計算が遅い**：CUDAカーネルなどのコンパイルが入ります。既定のキャッシュ先は `%LOCALAPPDATA%\GaussianCutSim\cuda-cache` です。
- **`[PERF] Log: ...` が表示される**：性能ログ `CutSim_performance.jsonl` の保存先を示す通常のメッセージです。
- **`CUDA path could not be detected` が表示される**：`setup.bat --check-only` で `CUDA kernel compilation and execution: OK` を確認してください。警告だけでは計算失敗を意味しません。
- **CUDA確認が失敗する**：エラー内容とGPUドライバーを確認してください。`--use-system-cuda` を使った場合はCUDA 12 Toolkitの設定を確認するか、オプションなしでセットアップを再実行します。
