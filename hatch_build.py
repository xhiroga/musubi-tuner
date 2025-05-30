import os
import re
import shutil
from pathlib import Path
from typing import Dict, List
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


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
            
            lines = content.splitlines()
            modified_lines = [
                self._rewrite_import_line(line, current_module_parts) 
                for line in lines
            ]
            
            file_path.write_text('\n'.join(modified_lines) + '\n', encoding='utf-8')
            
        except Exception as e:
            print(f"Warning: Failed to rewrite imports in {file_path}: {e}")
    
    def _rewrite_import_line(self, line: str, current_module_parts: List[str]) -> str:
        """Rewrite a single import line."""
        stripped_line = line.strip()
        
        if not stripped_line or stripped_line.startswith('#'):
            return line
        
        if re.match(r'^\s*from\s+\.', line):
            return self._rewrite_relative_import(line, current_module_parts)
        
        if self._is_module_import(line):
            return self._rewrite_absolute_import(line)
        
        if self._is_simple_module_import(line):
            return self._rewrite_simple_import(line)
        
        return line
    
    def _is_module_import(self, line: str) -> bool:
        """Check if line is an import of our modules."""
        pattern = r'^\s*from\s+(' + '|'.join(self.subpackages) + r')\b'
        return bool(re.match(pattern, line))
    
    def _is_simple_module_import(self, line: str) -> bool:
        """Check if line is a simple import of our modules."""
        pattern = r'^\s*import\s+(' + '|'.join(self.subpackages) + r')\b'
        return bool(re.match(pattern, line))
    
    def _rewrite_relative_import(self, line: str, current_module_parts: List[str]) -> str:
        """Rewrite relative imports to absolute imports."""
        match = re.match(r'^(\s*)from\s+(\.+)([^\s]*)\s+import\s+(.+)$', line)
        if not match:
            return line
        
        indent, dots, module_part, import_part = match.groups()
        level = len(dots)
        
        if level > len(current_module_parts):
            return line
        
        base_parts = current_module_parts[:-level+1] if level > 1 else current_module_parts
        
        if module_part:
            absolute_module = f"musubi_tuner.{'.'.join(base_parts + [module_part])}"
        else:
            absolute_module = f"musubi_tuner.{'.'.join(base_parts)}" if base_parts else "musubi_tuner"
        
        return f"{indent}from {absolute_module} import {import_part}"
    
    def _rewrite_absolute_import(self, line: str) -> str:
        """Rewrite absolute imports of our modules."""
        match = re.match(r'^(\s*)from\s+([^\s]+)\s+import\s+(.+)$', line)
        if not match:
            return line
        
        indent, module_name, import_part = match.groups()
        
        if module_name.split('.')[0] in self.subpackages:
            return f"{indent}from musubi_tuner.{module_name} import {import_part}"
        
        return line
    
    def _rewrite_simple_import(self, line: str) -> str:
        """Rewrite simple import statements."""
        match = re.match(r'^(\s*)import\s+([^\s]+)(.*)$', line)
        if not match:
            return line
        
        indent, module_name, rest = match.groups()
        
        if module_name.split('.')[0] in self.subpackages:
            return f"{indent}import musubi_tuner.{module_name}{rest}"
        
        return line
    
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