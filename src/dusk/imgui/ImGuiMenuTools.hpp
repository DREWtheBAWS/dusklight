#ifndef DUSK_IMGUI_MENUTOOLS_HPP
#define DUSK_IMGUI_MENUTOOLS_HPP

#include <aurora/aurora.h>
#include <queue>
#include <string>

#include "imgui.h"
#include "ImGuiSaveEditor.hpp"
#include "ImGuiStateShare.hpp"
#include "dusk/rtao/geometry_collector.hpp"
#include "dusk/rtao/blas_cache.hpp"
#include "dusk/rtao/tlas_builder.hpp"
#include "dusk/rtao/depth_viewer.hpp"
#include "dusk/rtao/ao_pass.hpp"
#include "dusk/rtao/gpu_bvh_builder.hpp"

namespace dusk {
    class ImGuiMenuTools {
    public:
        ImGuiMenuTools();
        void draw();
        void afterDraw();

		void ShowDebugOverlay();
		void ShowCameraOverlay();
		void ShowProcessManager();
		void ShowHeapOverlay();
		void ShowStubLog();
        void ShowBloomWindow();
        void ShowPlayerInfo();
        void ShowAudioDebug();
        void ShowSaveEditor();
        void ShowStateShare();
        void ShowInputViewer();
        void ShowActorSpawner();
        void ShowRtaoCaptureWindow();

    private:
		bool m_showDebugOverlay = false;
		int m_debugOverlayCorner = 2; // bottom-left

		bool m_showCameraOverlay = false;
		int m_cameraOverlayCorner = 3;

		bool m_showProcessManagement = false;

		bool m_showHeapOverlay = false;

		bool m_showStubLog = false;

        bool m_showBloomWindow = false;

        bool m_showAudioDebug = false;

		bool m_showPlayerInfo = false;
		int m_playerInfoOverlayCorner = 1; // top-right

		bool m_showSaveEditor = false;
        ImGuiSaveEditor m_saveEditor;

        bool m_showStateShare = false;
        ImGuiStateShare m_stateShare;

        bool m_showActorSpawner = false;
        int m_inputOverlayCorner = 3;
        std::string m_controllerName;

        bool m_showRtaoCapture  = false;
        bool m_buildBvhOnly    = false;  // debug: skip AO pass after BVH build
        bool m_bvhFrozen       = false;  // debug: stop rebuilding BVH each frame
        bool m_bvhCaptureOnce  = false;  // debug: do one rebuild then auto-freeze
        bool m_useTlasBvh      = false;  // false=LBVH path, true=BLAS/TLAS path
        dusk::rtao::BlasCache          m_blasCache;   // declared before m_collector (callback sets up pointer)
        dusk::rtao::TlasBuilder        m_tlasBuilder;
        dusk::rtao::GeometryCollector  m_collector;
        dusk::rtao::DepthTextureViewer m_depthViewer;
        dusk::rtao::GpuBvhBuilder      m_bvhBuilder;
        dusk::rtao::AoPass             m_aoPass;
    };
}

#endif  // DUSK_IMGUI_MENUTOOLS_HPP
