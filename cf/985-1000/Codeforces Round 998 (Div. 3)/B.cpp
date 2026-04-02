//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mp make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define endl '\n'
//using namespace std;
//int a[100005];
//int cow[2005][2005];
//void solve()
//{
//    int n, m;
//    cin >> n >> m;
//    int flag = 1;
//    for (int i = 1; i <= n; i++)
//    {
//        for (int j = 1; j <= m; j++)
//        {
//            cin >> cow[i][j];
//        }
//        sort(cow[i] + 1, cow[i] + 1 + m);
//    }
//    if (m > 1)
//    {
//        
//        for (int i = 1; i <= n; i++)
//        {
//            if (!flag)break;
//            for (int j = 1; j < m; j++)
//            {
//                if (j == 1)a[cow[i][j] + 1] = i;
//                if (!flag)break;
//                if (cow[i][j + 1] - cow[i][j] != n)
//                    flag = 0;
//            }
//        }
//    }
//    else
//    {
//        for (int i = 1; i <= n; i++)
//        {
//            a[cow[i][1] + 1] = i;
//        }
//    }
//    if (!flag)
//        cout << "-1" << endl;
//    else
//    {
//        for (int i = 1; i <= n; i++)
//            cout << a[i] << " ";
//        cout << endl;
//    }
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