
CREATE TABLE qwen.data_doc_store (
	id serial4 NOT NULL,
	"key" varchar NOT NULL,
	"namespace" varchar NOT NULL,
	value json NULL,
	CONSTRAINT "data_doc_store:unique_key_namespace" UNIQUE (key, namespace),
	CONSTRAINT data_doc_store_pkey PRIMARY KEY (id)
);
CREATE INDEX "data_doc_store:idx_key_namespace" ON qwen.data_doc_store USING btree (key, namespace);

drop table qwen.data_vec_store;

CREATE TABLE qwen.data_vec_store_3584 (
	id bigserial PRIMARY KEY,
	"text" varchar NOT NULL,
	"node_id" varchar NOT NULL,
	metadata_ json NULL,
	embedding vector(3584) null --it need to match the dim numbers of your embedding dim
);

CREATE TABLE qwen.data_idx_store (
	"key" varchar,
	"namespace" varchar,
	value json,
	primary key("key", "namespace")
	-- unique(key, namespace)
);