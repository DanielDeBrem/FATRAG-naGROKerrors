"""File management and browser API endpoints."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from datetime import datetime

from app.core.config import get_settings

router = APIRouter(prefix="/admin/files", tags=["files"])


def get_upload_directory() -> Path:
    """Get the configured upload directory."""
    settings = get_settings()
    upload_dir = Path(settings.upload_directory or "fatrag_data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file/directory information."""
    try:
        stat = file_path.stat()
        file_type = "file"
        mime_type = "application/octet-stream"

        if file_path.is_dir():
            file_type = "directory"
            mime_type = "inode/directory"
        elif file_path.is_file():
            # Try to guess MIME type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                # Detect based on extension
                ext = file_path.suffix.lower()
                if ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml']:
                    mime_type = "text/plain"
                elif ext in ['.pdf']:
                    mime_type = "application/pdf"
                elif ext in ['.docx']:
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif ext in ['.xlsx']:
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    mime_type = "image/" + ext[1:]

        return {
            "name": file_path.name,
            "path": str(file_path),
            "type": file_type,
            "mime_type": mime_type,
            "size": stat.st_size,
            "size_formatted": _format_file_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "modified_human": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "is_dir": file_path.is_dir(),
            "is_file": file_path.is_file(),
            "extension": file_path.suffix or "",
            "readable": os.access(file_path, os.R_OK),
            "writable": os.access(file_path, os.W_OK),
        }
    except Exception as e:
        return {
            "name": file_path.name,
            "path": str(file_path),
            "error": str(e),
            "type": "error"
        }


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return ".1f"
        size_bytes /= 1024.0
    return ".1f"


def _is_safe_path(base_path: Path, target_path: Path) -> bool:
    """Check if target path is within base path (prevent directory traversal)."""
    try:
        base_path = base_path.resolve()
        target_path = target_path.resolve()
        return target_path.exists() and base_path in target_path.parents or target_path == base_path
    except Exception:
        return False


@router.get("/", response_model=Dict[str, Any])
def list_files(
    path: str = Query("/", description="Directory path to list"),
    show_hidden: bool = Query(False, description="Show hidden files"),
    sort_by: str = Query("name", description="Sort by: name, size, modified"),
    sort_order: str = Query("asc", description="Sort order: asc, desc")
):
    """List files and directories in the specified path."""
    try:
        upload_dir = get_upload_directory()

        # Resolve the requested path
        if path == "/":
            target_dir = upload_dir
        else:
            target_dir = Path(path)
            if not target_dir.is_absolute():
                target_dir = upload_dir / path.strip("/")

        # Security check - ensure we're within the upload directory
        if not _is_safe_path(upload_dir, target_dir):
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")

        if not target_dir.exists():
            raise HTTPException(status_code=404, detail="Directory not found")

        if not target_dir.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        try:
            items = []
            for item in target_dir.iterdir():
                # Skip hidden files if not requested
                if not show_hidden and item.name.startswith('.'):
                    continue

                items.append(get_file_info(item))

            # Sort items
            reverse_order = sort_order.lower() == "desc"

            if sort_by == "size":
                items.sort(key=lambda x: x.get("size", 0), reverse=reverse_order)
            elif sort_by == "modified":
                items.sort(key=lambda x: x.get("modified", ""), reverse=reverse_order)
            else:  # name
                items.sort(key=lambda x: x.get("name", "").lower(), reverse=reverse_order)

            return {
                "path": str(target_dir),
                "relative_path": str(target_dir.relative_to(upload_dir)) if target_dir != upload_dir else "/",
                "items": items,
                "total_files": len([i for i in items if i.get("is_file")]),
                "total_dirs": len([i for i in items if i.get("is_dir")]),
                "total_size": sum(i.get("size", 0) for i in items if i.get("is_file")),
                "total_size_formatted": _format_file_size(sum(i.get("size", 0) for i in items if i.get("is_file")))
            }

        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing directory: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=Dict[str, Any])
def search_files(
    query: str = Query(..., description="Search query"),
    path: str = Query("/", description="Base path to search in"),
    file_types: Optional[List[str]] = Query(None, description="Filter by file extensions (e.g., .pdf,.txt)"),
    max_results: int = Query(100, description="Maximum number of results")
):
    """Search for files by name within the upload directory."""
    try:
        upload_dir = get_upload_directory()
        query_lower = query.lower()

        # Resolve search path
        if path == "/":
            search_dir = upload_dir
        else:
            search_dir = Path(path)
            if not search_dir.is_absolute():
                search_dir = upload_dir / path.strip("/")

        if not _is_safe_path(upload_dir, search_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not search_dir.exists():
            raise HTTPException(status_code=404, detail="Search path not found")

        results = []
        file_types_list = [ft.strip('.') for ft in (file_types or [])]

        def search_recursive(current_dir: Path):
            try:
                for item in current_dir.iterdir():
                    # Check file name match
                    if query_lower in item.name.lower():
                        # Check file type filter if specified
                        if file_types_list and item.is_file():
                            if item.suffix.strip('.') not in file_types_list:
                                continue

                        results.append(get_file_info(item))

                        if len(results) >= max_results:
                            return
                    elif item.is_dir():
                        search_recursive(item)
            except PermissionError:
                pass  # Skip directories we can't access

        search_recursive(search_dir)

        return {
            "query": query,
            "path": str(search_dir),
            "results": results,
            "total": len(results),
            "max_results": max_results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-folder")
def create_folder(
    name: str = Query(..., description="Folder name"),
    path: str = Query("/", description="Parent directory path")
):
    """Create a new folder."""
    try:
        upload_dir = get_upload_directory()

        # Resolve parent path
        if path == "/":
            parent_dir = upload_dir
        else:
            parent_dir = Path(path)
            if not parent_dir.is_absolute():
                parent_dir = upload_dir / path.strip("/")

        if not _is_safe_path(upload_dir, parent_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        # Create new folder
        new_dir = parent_dir / name.strip()

        if new_dir.exists():
            raise HTTPException(status_code=409, detail="Folder already exists")

        try:
            new_dir.mkdir(parents=True, exist_ok=False)
            return {"status": "created", "path": str(new_dir)}
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")

    except HTTPException:
        raise


@router.delete("/{file_path:path}")
def delete_file(file_path: str):
    """Delete a file or folder."""
    try:
        upload_dir = get_upload_directory()
        target_path = Path(file_path)

        if not target_path.is_absolute():
            target_path = upload_dir / file_path

        if not _is_safe_path(upload_dir, target_path):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            return {"status": "deleted", "path": file_path}
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting: {str(e)}")

    except HTTPException:
        raise


@router.post("/{file_path:path}/rename")
def rename_file(
    file_path: str,
    new_name: str = Query(..., description="New name")
):
    """Rename a file or folder."""
    try:
        upload_dir = get_upload_directory()
        target_path = Path(file_path)

        if not target_path.is_absolute():
            target_path = upload_dir / file_path

        if not _is_safe_path(upload_dir, target_path):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        new_path = target_path.parent / new_name.strip()

        if new_path.exists():
            raise HTTPException(status_code=409, detail="Destination already exists")

        try:
            target_path.rename(new_path)
            return {
                "status": "renamed",
                "old_path": file_path,
                "new_path": str(new_path.relative_to(upload_dir))
            }
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error renaming: {str(e)}")

    except HTTPException:
        raise


@router.get("/download/{file_path:path}")
def download_file(file_path: str):
    """Download a file."""
    try:
        upload_dir = get_upload_directory()
        target_path = Path(file_path)

        if not target_path.is_absolute():
            target_path = upload_dir / file_path

        if not _is_safe_path(upload_dir, target_path):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists() or not target_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=target_path,
            filename=target_path.name,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/preview/{file_path:path}")
def preview_file(file_path: str):
    """Get file preview information."""
    try:
        upload_dir = get_upload_directory()
        target_path = Path(file_path)

        if not target_path.is_absolute():
            target_path = upload_dir / file_path

        if not _is_safe_path(upload_dir, target_path):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists() or not target_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        file_info = get_file_info(target_path)

        # Add preview-specific information
        preview_data = file_info.copy()

        # For text files, include first few lines as preview
        if file_info["mime_type"] and file_info["mime_type"].startswith("text/") and file_info["size"] < 10240:  # 10KB
            try:
                with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(1000)  # First 1000 chars
                    preview_data["preview_content"] = content
                    preview_data["has_more"] = len(content) == 1000 and file_info["size"] > 1000
            except Exception:
                pass

        return preview_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting preview: {str(e)}")


@router.get("/stats")
def get_upload_stats():
    """Get upload directory statistics."""
    try:
        upload_dir = get_upload_directory()

        total_files = 0
        total_dirs = 0
        total_size = 0
        file_types = {}

        for root, dirs, files in os.walk(upload_dir):
            total_dirs += len(dirs)
            for file in files:
                total_files += 1
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                except:
                    pass

        return {
            "directory": str(upload_dir),
            "total_files": total_files,
            "total_dirs": total_dirs,
            "total_size": total_size,
            "total_size_formatted": _format_file_size(total_size),
            "file_types": file_types,
            "last_scanned": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
