import os
import re
import shutil
from pathlib import Path
from typing import Dict, List
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook to create musubi_tuner package structure with import rewriting."""
    
    PLUGIN_NAME = "custom"
    
    MODULES_TO_COPY = [
        "dataset", "frame_pack", "hunyuan_model", "modules", 
        "networks", "utils", "wan"
    ]
    
    ROOT_SCRIPTS = [
        "cache_latents.py", "cache_text_encoder_outputs.py", "convert_lora.py",
        "fpack_cache_latents.py", "fpack_cache_text_encoder_outputs.py", 
        "fpack_generate_video.py", "fpack_train_network.py",
        "hv_generate_video.py", "hv_train.py", "hv_train_network.py",
        "merge_lora.py", "wan_cache_latents.py", "wan_cache_text_encoder_outputs.py",
        "wan_generate_video.py", "wan_train_network.py"
    ]
    
    def initialize(self, version: str, build_data: Dict) -> None:
        """Initialize build process and create package structure."""
        self.source_dir = Path(self.root)
        self.package_name = "musubi_tuner"
        self.temp_dir = self.source_dir / "temp_build"
        self.package_dir = self.temp_dir / self.package_name
        
        self._setup_temp_directory()
        self._copy_files()
        self._rewrite_imports()
        self._move_to_final_location()
        
        build_data['packages'] = [self.package_name]
    
    def _setup_temp_directory(self) -> None:
        """Create clean temporary directory structure."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        self.package_dir.mkdir()
    
    def _copy_files(self) -> None:
        """Copy all necessary files to the package directory."""
        self._copy_modules()
        self._copy_root_scripts()
        self._create_init_file()
        self._ensure_init_files()
    
    def _copy_modules(self) -> None:
        """Copy module directories."""
        for module in self.MODULES_TO_COPY:
            src_path = self.source_dir / module
            if src_path.exists():
                shutil.copytree(src_path, self.package_dir / module)
    
    def _copy_root_scripts(self) -> None:
        """Copy root scripts."""
        for script in self.ROOT_SCRIPTS:
            src_path = self.source_dir / script
            if src_path.exists():
                shutil.copy2(src_path, self.package_dir / script)
    
    def _create_init_file(self) -> None:
        """Create main __init__.py file."""
        init_file = self.package_dir / "__init__.py"
        init_file.write_text('"""Musubi Tuner package."""\n')
    
    def _ensure_init_files(self) -> None:
        """Ensure all Python package directories have __init__.py files."""
        for root, _, files in os.walk(self.package_dir):
            if any(f.endswith('.py') for f in files):
                init_file = Path(root) / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("")
    
    def _rewrite_imports(self) -> None:
        """Rewrite all import statements to use musubi_tuner namespace."""
        for py_file in self._find_python_files():
            self._rewrite_file_imports(py_file)
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the package directory."""
        python_files = []
        for root, _, files in os.walk(self.package_dir):
            python_files.extend(Path(root) / file for file in files if file.endswith('.py'))
        return python_files
    
    def _rewrite_file_imports(self, file_path: Path) -> None:
        """Rewrite imports in a single Python file."""
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
        pattern = r'^\s*from\s+(' + '|'.join(self.MODULES_TO_COPY) + r')\b'
        return bool(re.match(pattern, line))
    
    def _is_simple_module_import(self, line: str) -> bool:
        """Check if line is a simple import of our modules."""
        pattern = r'^\s*import\s+(' + '|'.join(self.MODULES_TO_COPY) + r')\b'
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
        
        if module_name.split('.')[0] in self.MODULES_TO_COPY:
            return f"{indent}from musubi_tuner.{module_name} import {import_part}"
        
        return line
    
    def _rewrite_simple_import(self, line: str) -> str:
        """Rewrite simple import statements."""
        match = re.match(r'^(\s*)import\s+([^\s]+)(.*)$', line)
        if not match:
            return line
        
        indent, module_name, rest = match.groups()
        
        if module_name.split('.')[0] in self.MODULES_TO_COPY:
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