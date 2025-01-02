<script setup lang='ts'>
import { computed } from 'vue'
import { NLayout, NLayoutContent } from 'naive-ui'
import { useRouter } from 'vue-router'
import Sider from './sider/index.vue'
import { useBasicLayout } from '@/hooks/useBasicLayout'
import { useAppStore, useChatStore } from '@/store'

const router = useRouter()
const appStore = useAppStore()
const chatStore = useChatStore()

router.replace({ name: 'Chat', params: { uuid: chatStore.active } })

const { isMobile } = useBasicLayout()

const collapsed = computed(() => appStore.siderCollapsed)

const getMobileClass = computed(() => {
  if (isMobile.value)
    return ['rounded-none', 'shadow-none']
  return ['rounded-3xl', 'shadow-lg', 'dark:border-neutral-800']
})

const getContainerClass = computed(() => {
  return [
    'h-full',
    { 'pl-[260px]': !isMobile.value && !collapsed.value },
  ]
})
</script>

<template>
  <NLayout class="z-40 transition" :class="getContainerClass" has-sider>
    <Sider />
    <NLayoutContent class="h-full">
      <div class="h-full transition-all bg-gradient-to-bl from-[#dde9ff] to-[#eae4fc]" :class="[isMobile ? 'p-0' : 'p-4']">
        <div class="h-full overflow-hidden bg-white/50 backdrop-blur-sm" :class="getMobileClass">
          <RouterView v-slot="{ Component, route }">
            <component :is="Component" :key="route.fullPath" />
          </RouterView>
        </div>
      </div>
    </NLayoutContent>
  </NLayout>
</template>
