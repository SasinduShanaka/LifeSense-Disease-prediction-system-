"""
File and Directory Operations Utilities
"""
import os
import shutil
import json
import csv
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime


class FileManager:
    """Handle file operations"""
    
    @staticmethod
    def create_directory(path):
        """Create directory if it doesn't exist"""
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def list_files(directory, extension=None, recursive=False):
        """List files in directory"""
        path = Path(directory)
        
        if recursive:
            pattern = '**/*' if not extension else f'**/*{extension}'
        else:
            pattern = '*' if not extension else f'*{extension}'
        
        return [str(f) for f in path.glob(pattern) if f.is_file()]
    
    @staticmethod
    def copy_file(source, destination):
        """Copy file to destination"""
        shutil.copy2(source, destination)
        return destination
    
    @staticmethod
    def move_file(source, destination):
        """Move file to destination"""
        shutil.move(source, destination)
        return destination
    
    @staticmethod
    def delete_file(path):
        """Delete file"""
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    
    @staticmethod
    def delete_directory(path):
        """Delete directory and its contents"""
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False
    
    @staticmethod
    def get_file_size(path):
        """Get file size in bytes"""
        return os.path.getsize(path)
    
    @staticmethod
    def get_file_info(path):
        """Get file metadata"""
        stat = os.stat(path)
        return {
            'path': path,
            'name': os.path.basename(path),
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': os.path.isfile(path),
            'is_dir': os.path.isdir(path)
        }
    
    @staticmethod
    def rename_file(old_path, new_name):
        """Rename file"""
        directory = os.path.dirname(old_path)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        return new_path
    
    @staticmethod
    def read_text_file(path, encoding='utf-8'):
        """Read text file"""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text_file(path, content, encoding='utf-8'):
        """Write text file"""
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def append_to_file(path, content):
        """Append content to file"""
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def read_json(path):
        """Read JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(path, data, indent=2):
        """Write JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def read_csv(path):
        """Read CSV file"""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    @staticmethod
    def write_csv(path, data, fieldnames):
        """Write CSV file"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    @staticmethod
    def calculate_checksum(path, algorithm='md5'):
        """Calculate file checksum"""
        hash_func = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def find_duplicates(directory):
        """Find duplicate files in directory"""
        files_by_hash = {}
        
        for file_path in FileManager.list_files(directory, recursive=True):
            file_hash = FileManager.calculate_checksum(file_path)
            
            if file_hash in files_by_hash:
                files_by_hash[file_hash].append(file_path)
            else:
                files_by_hash[file_hash] = [file_path]
        
        # Return only duplicates
        return {k: v for k, v in files_by_hash.items() if len(v) > 1}


class ArchiveManager:
    """Handle archive operations"""
    
    @staticmethod
    def create_zip(source_dir, output_path):
        """Create ZIP archive"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        
        return output_path
    
    @staticmethod
    def extract_zip(zip_path, extract_to):
        """Extract ZIP archive"""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
        
        return extract_to
    
    @staticmethod
    def list_zip_contents(zip_path):
        """List contents of ZIP archive"""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            return zipf.namelist()
    
    @staticmethod
    def add_to_zip(zip_path, file_path, arcname=None):
        """Add file to existing ZIP"""
        with zipfile.ZipFile(zip_path, 'a') as zipf:
            zipf.write(file_path, arcname or os.path.basename(file_path))


class DirectoryOrganizer:
    """Organize files in directories"""
    
    @staticmethod
    def organize_by_extension(source_dir, target_dir):
        """Organize files by extension into subdirectories"""
        FileManager.create_directory(target_dir)
        
        for file_path in FileManager.list_files(source_dir):
            ext = Path(file_path).suffix[1:] or 'no_extension'
            ext_dir = os.path.join(target_dir, ext)
            
            FileManager.create_directory(ext_dir)
            
            dest_path = os.path.join(ext_dir, os.path.basename(file_path))
            FileManager.copy_file(file_path, dest_path)
    
    @staticmethod
    def organize_by_date(source_dir, target_dir):
        """Organize files by modification date"""
        FileManager.create_directory(target_dir)
        
        for file_path in FileManager.list_files(source_dir):
            mtime = os.path.getmtime(file_path)
            date = datetime.fromtimestamp(mtime)
            
            date_dir = os.path.join(target_dir, date.strftime('%Y'), date.strftime('%m'))
            FileManager.create_directory(date_dir)
            
            dest_path = os.path.join(date_dir, os.path.basename(file_path))
            FileManager.copy_file(file_path, dest_path)
    
    @staticmethod
    def organize_by_size(source_dir, target_dir, size_ranges=None):
        """Organize files by size ranges"""
        if size_ranges is None:
            size_ranges = {
                'small': (0, 1024 * 1024),  # < 1MB
                'medium': (1024 * 1024, 10 * 1024 * 1024),  # 1MB - 10MB
                'large': (10 * 1024 * 1024, float('inf'))  # > 10MB
            }
        
        FileManager.create_directory(target_dir)
        
        for file_path in FileManager.list_files(source_dir):
            size = FileManager.get_file_size(file_path)
            
            category = None
            for cat_name, (min_size, max_size) in size_ranges.items():
                if min_size <= size < max_size:
                    category = cat_name
                    break
            
            if category:
                cat_dir = os.path.join(target_dir, category)
                FileManager.create_directory(cat_dir)
                
                dest_path = os.path.join(cat_dir, os.path.basename(file_path))
                FileManager.copy_file(file_path, dest_path)


def batch_rename(directory, pattern, replacement):
    """Batch rename files matching pattern"""
    renamed = []
    
    for file_path in FileManager.list_files(directory):
        filename = os.path.basename(file_path)
        
        if pattern in filename:
            new_name = filename.replace(pattern, replacement)
            new_path = FileManager.rename_file(file_path, new_name)
            renamed.append((file_path, new_path))
    
    return renamed


def clean_empty_directories(root_dir):
    """Remove empty directories recursively"""
    removed = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            removed.append(dirpath)
    
    return removed
