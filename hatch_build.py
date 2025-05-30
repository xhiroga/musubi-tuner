import os
import shutil
import ast
from pathlib import Path
from typing import Dict, List
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


"""
Hook for bundling musubi-tuner modules into the `musubi_tuner` package.

Test procedure:
1. `rm -rf dist && rm -rf build_test`
2. `uvx hatch build`
3. `unzip dist/musubi_tuner-*.whl -d dist/whl/`
4. `tar -xzf dist/musubi_tuner-*.tar.gz -C dist/`
5. `mkdir build_test && uv --directory build_test init --no-workspace`
6. `uv --project build-test add .`
7. `uv --project build-test run python -c "import musubi_tuner.fpack_cache_latents"`
"""

class RelativeToAbsoluteTransformer(ast.NodeTransformer):
    def __init__(self, package_name: str, current_module_parts: List[str], subpackages: List[str], root_scripts: List[str]):
        self.package_name = package_name
        self.current_module_parts = current_module_parts
        self.subpackages = subpackages
        self.root_scripts = root_scripts
        # root_scriptsから.pyを除いたモジュール名のセットを作成
        self.root_script_modules = {script[:-3] for script in root_scripts if script.endswith('.py')}

    def _is_our_module(self, module_name: str) -> bool:
        """Check if a module name is one of our subpackages or root scripts."""
        if not module_name:
            return False
        
        base_module = module_name.split('.')[0]
        return base_module in self.subpackages or base_module in self.root_script_modules

    def _convert_relative_to_absolute(self, module_name: str, level: int) -> str:
        """Convert relative import to absolute import path."""
        # 現在のモジュールから相対的な位置を計算
        if level > len(self.current_module_parts):
            # レベルが深すぎる場合は変換しない
            return module_name
        
        # 基準となるモジュール部分を計算
        base_parts = self.current_module_parts[:-level+1] if level > 1 else self.current_module_parts
        
        if module_name:
            # from .module import something の場合
            return f"{self.package_name}.{'.'.join(base_parts + [module_name])}"
        else:
            # from . import something の場合
            return f"{self.package_name}.{'.'.join(base_parts)}" if base_parts else self.package_name

    def _add_package_prefix(self, module_name: str) -> str:
        """Add package prefix to our module."""
        return f"{self.package_name}.{module_name}"

    def visit_ImportFrom(self, node):
        # 相対インポートの処理
        if node.level and node.level > 0:
            abs_module = self._convert_relative_to_absolute(node.module, node.level)
            if abs_module != node.module:  # 変換が行われた場合のみ
                print(f"[TRANSFORM] from {'.' * node.level}{node.module or ''} -> from {abs_module}")
                return ast.ImportFrom(module=abs_module, names=node.names, level=0)
        
        # サブパッケージやroot_scriptsの絶対インポートの処理
        elif self._is_our_module(node.module):
            abs_module = self._add_package_prefix(node.module)
            print(f"[TRANSFORM] from {node.module} -> from {abs_module}")
            return ast.ImportFrom(module=abs_module, names=node.names, level=0)
        
        return node

    def visit_Import(self, node):
        """Handle import statements for subpackages and root scripts."""
        new_names = []
        has_changes = False
        
        for alias in node.names:
            if self._is_our_module(alias.name):
                new_module_name = self._add_package_prefix(alias.name)
                print(f"[TRANSFORM] import {alias.name} -> import {new_module_name}")
                new_alias = ast.alias(name=new_module_name, asname=alias.asname)
                new_names.append(new_alias)
                has_changes = True
            else:
                new_names.append(alias)
        
        return ast.Import(names=new_names) if has_changes else node


class CustomBuildHook(BuildHookInterface):
    """Build hook to create musubi_tuner package structure with import rewriting."""
    
    PLUGIN_NAME = "custom"
    
    def initialize(self, version: str, build_data: Dict) -> None:
        """Initialize build process and create package structure."""
        self.source_dir = Path(self.root)
        self.package_name = "musubi_tuner"
        self.temp_dir = self.source_dir / "temp"
        self.package_dir = self.temp_dir / self.package_name
        
        self.subpackages = self._discover_subpackages()
        self.root_scripts = self._discover_root_scripts()
        
        self._setup_temp_directory()
        self._copy_files()
        self._rewrite_imports()
        self._move_to_final_location()
        
        build_data['packages'] = [self.package_name]
    
    def _discover_subpackages(self) -> List[str]:
        subpackages = []
        
        for item in self.source_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('_'):
                init_file = item / "__init__.py"
                if init_file.exists():
                    subpackages.append(item.name)
        
        subpackages.sort()
        return subpackages
    
    def _discover_root_scripts(self) -> List[str]:
        exclude_patterns = {'hatch_build.py'}
        
        root_scripts = []
        
        for item in self.source_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name not in exclude_patterns:
                root_scripts.append(item.name)
        
        root_scripts.sort()
        return root_scripts
    
    def _setup_temp_directory(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        self.package_dir.mkdir()
    
    def _copy_files(self) -> None:
        self._copy_subpackages()
        self._copy_root_scripts()
        self._create_init_file()
    
    def _copy_subpackages(self) -> None:
        for subpackage in self.subpackages:
            src_path = self.source_dir / subpackage
            if src_path.exists():
                shutil.copytree(src_path, self.package_dir / subpackage)
    
    def _copy_root_scripts(self) -> None:
        for script in self.root_scripts:
            src_path = self.source_dir / script
            if src_path.exists():
                shutil.copy2(src_path, self.package_dir / script)
    
    def _create_init_file(self) -> None:
        init_file = self.package_dir / "__init__.py"
        init_file.write_text('"""Musubi Tuner package."""\n')
    
    def _rewrite_imports(self) -> None:
        for root, _, files in os.walk(self.package_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._rewrite_file_imports(file_path)
    
    def _rewrite_file_imports(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding='utf-8')
            relative_path = file_path.relative_to(self.package_dir)
            current_module_parts = list(relative_path.parent.parts)
            
            # ASTを使用してファイルを解析
            tree = ast.parse(content)
            
            # 相対インポートを絶対インポートに変換
            transformer = RelativeToAbsoluteTransformer(self.package_name, current_module_parts, self.subpackages, self.root_scripts)
            new_tree = transformer.visit(tree)
            
            # 変更されたASTをコードに戻す
            new_content = ast.unparse(new_tree)
            file_path.write_text(new_content, encoding='utf-8')
            
        except Exception as e:
            print(f"Warning: Failed to rewrite imports in {file_path}: {e}")
    
    def _move_to_final_location(self) -> None:
        """Move package to root level for proper wheel structure."""
        final_package_dir = self.source_dir / self.package_name
        if final_package_dir.exists():
            shutil.rmtree(final_package_dir)
        shutil.move(str(self.package_dir), str(final_package_dir))
    
    def finalize(self, version: str, build_data: Dict, artifact_path: str) -> None:
        """Clean up temporary files after build."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        final_package_dir = self.source_dir / self.package_name
        if final_package_dir.exists():
            shutil.rmtree(final_package_dir)