"""Central model hub with canonical backbone/aggregator configs.

These entries define the *authoritative* model hyper-parameters. When a
`--weights` key references a hub entry, its backbone/aggregator settings
must override user/YAML values.
"""

from __future__ import annotations

MODEL_HUBS = {
	"resnet50_avg": {
		"backbone": "resnet50",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"resnet101_avg": {
		"backbone": "resnet101",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
    "resnet50_gem": {
        "backbone": "resnet50",
        "aggregator": "gem",
        "backbone_args": {    
            "unfreeze_n_blocks": 1, 
			"pretrained": True,
			"crop_last_block": False},
        "aggregator_args": {"p": 3.0},
        "use_bnneck": True,
		"whitening": False,
		"whitening_dim": 1024,
    },
    "swinv2b_avg": {
		"backbone": "swin_v2b",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"swinv2l_avg": {
		"backbone": "swin_v2l",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2b14_cls": {
		"backbone": "dinov2_vitb14",
		"aggregator": "cls",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2b14_avg": {
		"backbone": "dinov2_vitb14",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2b14_reg_cls": {
		"backbone": "dinov2_vitb14_reg",
		"aggregator": "cls",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2g14_l31_avg": {
		"backbone": "dinov2_vitg14",
		"aggregator": "avg",
		"backbone_args": {"num_layers": 31},
		"aggregator_args": {},
	},
	"dinov3b16_cls": {
		"backbone": "dinov3_vitb16",
		"aggregator": "cls",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov3b16_avg": {
		"backbone": "dinov3_vitb16",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2b14_netvlad": {
		"backbone": "dinov2_vitb14",
		"aggregator": "netvlad",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"dinov2b14_regvpr": {
		"backbone": "dinov2_vitb14",
		"aggregator": "regvpr",
		"backbone_args": {},
		"aggregator_args": {
			"num_register_tokens": 4,
			"num_layers": 2,
			"num_heads": 12,
			"mlp_ratio": 4.0,
			"dropout": 0.0,
		},
	},
    "convb_avg": {
		"backbone": "dinov3_convnext_base",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
	"convl_avg": {
		"backbone": "dinov3_convnext_large",
		"aggregator": "avg",
		"backbone_args": {},
		"aggregator_args": {},
	},
    
# =============================== On the shelf Models ===============================
	"resnet50_cosplace": {
		"backbone": "resnet50",
		"aggregator": "cosplace",
		"backbone_args": {"crop_last_block": False},
		"aggregator_args": {"out_channels": 512},
	},
	"resnet50_boq_16384": {
		"backbone": "resnet50",
		"aggregator": "boq",
		"backbone_args": {"crop_last_block": True},
		"aggregator_args": {
			"proj_channels": 512,
			"num_queries": 64,
			"num_layers": 2,
			"row_dim": 32,
		},
	},
	"dinov2_boq_12288": {
		"backbone": "dinov2_vitb14",
		"aggregator": "boq",
		"backbone_args": {},
		"aggregator_args": {
			"proj_channels": 384,
			"num_queries": 64,
			"num_layers": 2,
			"row_dim": 32,
		},
	},
	"dino_salad": {
		"backbone": "dinov2_vitb14",
		"aggregator": "salad",
		"backbone_args": {},
		"aggregator_args": {
			"num_clusters": 64,
			"cluster_dim": 128,
			"token_dim": 256,
		},
	},
}

