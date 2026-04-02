// Created on: 2025-09-02 15:01:25

#include <bits/stdc++.h>
#define int long long
using namespace std;
using i64 = long long;

#if !defined(ONLINE_JUDGE) && defined(LOCAL)
// #if !defined(ONLINE_JUDGE)
#include "D:\VSC P\ide\cpp\cpphead\helper.h"
#else
#define dbg(...) ;
#define local_go_m(x) \
    int c;            \
    cin >> c;         \
    while (c--)       \
    x()
#define local_go(x) x()
#endif

void go()
{
    int n;
    cin >> n;
    vector<int> a(n + 2, INT_MAX);
    for (int i = 1; i <= n; i++)
        cin >> a[i];

    vector<int> st;
    int ans = 0;
    st.push_back(0);
    for (int i = 1; i <= n; i++)
    {
        while (!st.empty() && a[st.back()] < a[i])
            st.pop_back();
        ans += i - st.back() - 1;
        // cout << ans << endl;
        st.push_back(i);
    }
    while (!st.empty())
        st.pop_back();
    st.push_back(n + 1);
    for (int i = n; i >= 1; i--)
    {
        while (!st.empty() && a[st.back()] < a[i])
            st.pop_back();
        if (a[st.back()] != a[i])
            ans += st.back() - i - 1;
        // cout << ans << endl;
        st.push_back(i);
    }
    cout << ans << endl;
    // cout << endl;
}

signed main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    local_go_m(go);
    // local_go(go);
    return 0;
}