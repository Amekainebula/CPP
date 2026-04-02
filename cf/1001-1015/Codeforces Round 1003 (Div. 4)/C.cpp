//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mpr make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define INF 0x7fffffffffffffff
//#define endl '\n'
//using namespace std;
//void solve()
//{
//    /*int n, m;
//    cin >> n >> m;
//    for (int i = 1; i <= n; i++)
//        cin >> a[i];
//    for (int i = 1; i <= m; i++)
//        cin >> b[i];
//    for (int i = 1; i <= n; i++)
//    {
//        c[i] = b[1] - a[i];
//    }
//    a[1] = min(a[1], c[1]);
//    for (int i = 2; i <= n; i++)
//    {
//        int x = 1e9, y = 1e9;
//        if (a[i] >= a[i - 1])x = a[i];
//        if (c[i] >= a[i - 1])y = c[i];
//        if (x == 1e9 && y == 1e9)
//        {
//            cout << "NO" << endl;
//            return;
//        }
//        a[i] = min(x, y);
//    }
//    cout << "YES" << endl;*/
//    int n, m;
//    cin >> n >> m;
//    vector<int> a(n), b(m), c(n);
//    for (int i = 0; i < n; i++)
//    {
//        cin >> a[i];
//    }
//    int minb = 1e9;
//    for (int i = 0; i < m; i++)
//    {
//        cin >> b[i];
//        minb = min(minb, b[i]);
//    }
//    sort(all(b));
//    a[0] = min(a[0], minb - a[0]);
//    for (int i = 1; i < n; i++)
//    {
//        auto it = lower_bound(all(b), a[i - 1] + a[i]);
//        if (it == b.end())
//        {
//            if (a[i] < a[i - 1])
//            {
//                cout << "NO" << endl;
//                return;
//            }
//            continue;
//        }
//        int j = b[it - b.begin()];
//        if (a[i] < a[i - 1])
//        {
//            a[i] = j - a[i];
//        }
//        else
//        {
//            a[i] = min(a[i], j - a[i]);
//        }
//    }
//    cout << "YES" << endl;
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}