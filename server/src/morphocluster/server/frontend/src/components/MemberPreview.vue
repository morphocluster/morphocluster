<template>
    <div class="member-preview card" :style="style">
        <div class="member-header" :title="title">
            <div class="member-title" v-if="show_title">{{ title }}</div>
            <div class="member-controls">
                <i
                    v-for="c of controls"
                    :key="c.event"
                    class="mdi mdi-dark"
                    :class="c.icon"
                    :title="c.title"
                    v-on:click="$emit(c.event, member)"
                />
            </div>
        </div>
        <div class="member-body" :class="{ oneImage: image_urls.length == 1 }">
            <img v-for="img_url of image_urls" :src="img_url" :key="img_url" />
        </div>
    </div>
</template>

<script>
import "@mdi/font/css/materialdesignicons.css";

export default {
    name: "member-preview",
    props: ["member", "controls"],
    data() {
        return {
            show_title: !!window.config.FRONTEND_SHOW_MEMBER_TITLE,
        };
    },
    methods: {},
    mounted() {},
    computed: {
        title: function () {
            return "object_id" in this.member
                ? this.member.object_id
                : this.member.node_id;
        },
        style: function () {
            return {
                backgroundColor:
                    "object_id" in this.member ? "#a6cee3" : "#fdbf6f",
            };
        },
        image_urls: function () {
            if ("object_id" in this.member) {
                return [`/get_obj_image/${this.member.object_id}`];
            } else if ("type_objects" in this.member) {
                return this.member.type_objects.map(
                    (objid) => `/get_obj_image/${objid}`
                );
            }
            return [];
        },
    },
};
</script>

<style scoped>
.member-preview {
    margin-bottom: 15px;
    color: black;
}

.member-header {
    border-top-right-radius: calc(0.25rem - 1px);
    border-top-left-radius: calc(0.25rem - 1px);
    padding-top: 3px;
    height: auto;
    /* display: flex; */
}

.member-body {
    /*background-color: white;*/
    background-color: var(--background-color);
}

.member-title {
    flex-grow: 1;
    overflow: hidden;
    /*
	white-space: nowrap;
	text-overflow: ellipsis;
	*/
}

.member-body img {
    width: 33.333333%;
}

.member-body.oneImage img {
    max-width: 100%;
    width: unset;
    max-height: 150px;
    margin: 0 auto;
    display: block;
}

.member-controls {
    font-size: x-large;
    /* margin-left: auto; */
    float: right;
}

.member-controls i {
    cursor: pointer;
    margin-left: 0.25em;
}
</style>