"""
Cliente para armazenamento de objetos (MinIO).

PR3 v2 - Hard Reset RAG Architecture

Layout de keys no bucket:
- documents/{document_id}/source.pdf
- documents/{document_id}/canonical.md
- documents/{document_id}/manifest.json
"""

import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error


@dataclass
class ObjectStat:
    """Estatísticas de um objeto no storage."""

    key: str
    size: int
    content_type: str
    etag: str
    last_modified: str


class ObjectStorageClient:
    """
    Cliente para MinIO/S3.

    Usa o container MinIO existente na VPS (usado para imagens do blog).
    NÃO usar o MinIO do Milvus.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: bool = False,
    ):
        """
        Inicializa o cliente MinIO.

        Args:
            endpoint: URL do MinIO (default: env MINIO_ENDPOINT)
            access_key: Access key (default: env MINIO_ACCESS_KEY)
            secret_key: Secret key (default: env MINIO_SECRET_KEY)
            bucket: Nome do bucket (default: env MINIO_BUCKET ou "rag-documents")
            secure: Usar HTTPS (default: False para conexão local)
        """
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "rag-documents")
        self.secure = secure

        self._client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        # Garante que o bucket existe
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Cria o bucket se não existir."""
        try:
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
        except S3Error as e:
            raise RuntimeError(f"Erro ao criar/verificar bucket {self.bucket}: {e}")

    # =========================================================================
    # Operações principais
    # =========================================================================

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Armazena bytes no storage.

        Args:
            key: Chave do objeto (ex: "documents/LEI-14133-2021/source.pdf")
            data: Bytes a armazenar
            content_type: MIME type do conteúdo

        Returns:
            ETag do objeto armazenado
        """
        from io import BytesIO

        stream = BytesIO(data)
        result = self._client.put_object(
            bucket_name=self.bucket,
            object_name=key,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        return result.etag

    def get_bytes(self, key: str) -> bytes:
        """
        Obtém bytes do storage.

        Args:
            key: Chave do objeto

        Returns:
            Bytes do objeto

        Raises:
            FileNotFoundError: Se objeto não existir
        """
        try:
            response = self._client.get_object(self.bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"Objeto não encontrado: {key}")
            raise

    def presign_get_url(
        self,
        key: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """
        Gera URL pré-assinada para download.

        Args:
            key: Chave do objeto
            expires: Tempo de validade da URL

        Returns:
            URL pré-assinada
        """
        return self._client.presigned_get_object(
            bucket_name=self.bucket,
            object_name=key,
            expires=expires,
        )

    def stat(self, key: str) -> Optional[ObjectStat]:
        """
        Obtém estatísticas de um objeto.

        Args:
            key: Chave do objeto

        Returns:
            ObjectStat ou None se não existir
        """
        try:
            obj = self._client.stat_object(self.bucket, key)
            return ObjectStat(
                key=key,
                size=obj.size,
                content_type=obj.content_type,
                etag=obj.etag,
                last_modified=obj.last_modified.isoformat() if obj.last_modified else "",
            )
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    def exists(self, key: str) -> bool:
        """Verifica se um objeto existe."""
        return self.stat(key) is not None

    def delete(self, key: str) -> bool:
        """
        Remove um objeto.

        Args:
            key: Chave do objeto

        Returns:
            True se removido, False se não existia
        """
        try:
            self._client.remove_object(self.bucket, key)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    # =========================================================================
    # Helpers para layout de documentos
    # =========================================================================

    @staticmethod
    def source_key(document_id: str) -> str:
        """Retorna a key para o PDF fonte."""
        return f"documents/{document_id}/source.pdf"

    @staticmethod
    def canonical_key(document_id: str) -> str:
        """Retorna a key para o markdown canônico."""
        return f"documents/{document_id}/canonical.md"

    @staticmethod
    def manifest_key(document_id: str) -> str:
        """Retorna a key para o manifest."""
        return f"documents/{document_id}/manifest.json"

    def put_source_pdf(self, document_id: str, pdf_bytes: bytes) -> str:
        """Armazena o PDF fonte e retorna o SHA256."""
        key = self.source_key(document_id)
        self.put_bytes(key, pdf_bytes, content_type="application/pdf")
        return hashlib.sha256(pdf_bytes).hexdigest()

    def put_canonical_md(self, document_id: str, md_content: str) -> str:
        """Armazena o markdown canônico e retorna o SHA256."""
        key = self.canonical_key(document_id)
        md_bytes = md_content.encode("utf-8")
        self.put_bytes(key, md_bytes, content_type="text/markdown; charset=utf-8")
        return hashlib.sha256(md_bytes).hexdigest()

    def put_manifest(self, document_id: str, manifest_json: str) -> str:
        """Armazena o manifest JSON."""
        key = self.manifest_key(document_id)
        return self.put_bytes(
            key,
            manifest_json.encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def get_canonical_md(self, document_id: str) -> str:
        """Obtém o markdown canônico."""
        key = self.canonical_key(document_id)
        return self.get_bytes(key).decode("utf-8")
