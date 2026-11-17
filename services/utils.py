import json
import hashlib
import hmac
import datetime


class Signer:
    def __init__(self, request_data, service, region='cn-north-1'):
        self.method = request_data.get('method', 'POST').upper()
        self.path = request_data.get('path', '/')
        self.params = request_data.get('params', {})
        self.headers = request_data.get('headers', {})
        self.body = request_data.get('body', {})
        self.service = service
        self.region = region

    def add_authorization(self, account_config):
        ak = account_config.get('accessKeyId')
        sk = account_config.get('secretKey')
        if not ak or not sk:
            return

        now = datetime.datetime.utcnow()
        date = now.strftime("%Y%m%d")
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        self.headers['X-Date'] = ts

        body_str = json.dumps(self.body, separators=(",", ":"), ensure_ascii=False) if self.body else ''

        body_hash = hashlib.sha256(body_str.encode('utf-8')).hexdigest()
        self.headers['X-Content-Sha256'] = body_hash

        signed_headers = sorted([k.lower() for k in self.headers.keys() if k.lower() in ['content-type', 'host', 'x-content-sha256', 'x-date']])
        canonical_headers = "".join([f"{k}:{self.headers.get(key_map(k, self.headers)).strip()}\n" for k in signed_headers])
        signed_headers_str = ";".join(signed_headers)

        query_str = "&".join([f"{k}={v}" for k, v in sorted(self.params.items())])

        canonical_request = f"{self.method}\n{self.path}\n{query_str}\n{canonical_headers}\n{signed_headers_str}\n{body_hash}"

        credential_scope = f"{date}/{self.region}/{self.service}/request"
        string_to_sign = f"HMAC-SHA256\n{ts}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

        k_date = hmac_sha256(sk.encode('utf-8'), date)
        k_region = hmac_sha256(k_date, self.region)
        k_service = hmac_sha256(k_region, self.service)
        k_signing = hmac_sha256(k_service, "request")

        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        auth_header = f"HMAC-SHA256 Credential={ak}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        self.headers['Authorization'] = auth_header


def hmac_sha256(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def key_map(lower_key, headers):
    for k in headers.keys():
        if k.lower() == lower_key:
            return k
    return lower_key
