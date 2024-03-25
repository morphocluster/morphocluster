<template>
    <div class="member-preview card" :style="style">
        <div class="member-header" :title="title">
            <div class="member-title" v-if="show_title">{{ title }}</div>
            <div class="member-controls">
                <i v-for="c of controls" :key="c.event" class="mdi mdi-dark" :class="c.icon" :title="c.title"
                    v-on:click="$emit(c.event, member)" />
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
    mounted() { },
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
.member-title {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.smaller-title {
    font-size: 0.8em;
}

.member-preview {
    margin-bottom: 15px;
    max-height: 200px;
    color: black;
    border: 1px solid #2196F3;
    border-radius: 4px;
}

.member-header {
    background-color: #90caf9;
    border-top-right-radius: 4px;
    border-top-left-radius: 4px;
    padding: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.member-body {
    background-color: rgb(255, 255, 255);
}

.member-body img {
    max-width: 90%;
    max-height: 90%;
    max-height: 150px;
    width: auto;
    height: auto;
    margin-right: 2px;
}

.member-controls {
    font-size: x-large;
}

.member-controls i {
    cursor: pointer;
    margin-left: 0.25em;
}
</style>
