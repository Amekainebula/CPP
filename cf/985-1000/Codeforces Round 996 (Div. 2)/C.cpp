//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define ll long long
//#define ld long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//ll sumr[1005];
//ll sumrr[1005];
//ll suml[1005];
//ll sumll[1005];
//ll maps[1005][1005];
//ll mapss[1005][1005];
//queue<pair<ll, ll>> q;
//queue<pair<ll, ll>> qq;
//string s;
//ll n, m;
//void ss();
//void solve()
//{
//    memset(sumr, 0, sizeof(sumr));
//    memset(sumrr, 0, sizeof(sumrr));
//    memset(suml, 0, sizeof(suml));
//    memset(sumll, 0, sizeof(sumll));
//    while (!q.empty())q.pop();
//    while (!qq.empty())qq.pop();
//    ll maxx = 0;
//    cin >> n >> m;
//    cin >> s;
//    ll x = 1, y = 1;
//    q.push({ x, y });
//    qq.push({ x, y });
//    for (ll i = 0; i < s.length(); i++)
//    {
//        if (s[i] == 'D')
//            x++;
//        else
//            y++;
//        q.push({ x, y });
//        qq.push({ x, y });
//    }
//    for (ll i = 1; i <= n; i++)
//    {
//        for (ll j = 1; j <= m; j++)
//        {
//            cin >> maps[i][j];
//            mapss[i][j] = maps[i][j];
//            sumr[i] += maps[i][j];
//            suml[j] += maps[i][j];
//            sumrr[i] = sumr[i];
//            sumll[j] = suml[j];
//        }
//    }
//    for (ll i = 1; i <= n; i++)
//    {
//        if (sumr[i] > maxx)
//            maxx = sumr[i];
//        if (suml[i] > maxx)
//            maxx = suml[i];
//    }
//    int ff = 0;
//    while (!q.empty())
//    {
//        auto p = q.front();
//        q.pop();
//        if (ff < s.length())
//        {
//            if (s[ff] == 'D')
//            {
//                maps[p.first][p.second] = maxx - sumr[p.first];
//                sumr[p.first] = maxx;
//                suml[p.second] += maps[p.first][p.second];
//            }
//            else if (s[ff] == 'R')
//            {
//                maps[p.first][p.second] = maxx - suml[p.second];
//                suml[p.second] = maxx;
//                sumr[p.first] += maps[p.first][p.second];
//            }
//            ff++;
//        }
//        else
//        {
//            if (s[ff - 1] == 'D')
//            {
//                maps[p.first][p.second] = maxx - suml[p.second];
//                suml[p.second] = maxx;
//                sumr[p.first] += maps[p.first][p.second];
//            }
//            else if (s[ff - 1] == 'R')
//            {
//                maps[p.first][p.second] = maxx - sumr[p.first];
//                sumr[p.first] = maxx;
//                suml[p.second] += maps[p.first][p.second];
//            }
//        }
//    }
//    bool flag = true;
//    for (ll i = 1; i <= n; i++)
//    {
//        if (sumr[i] != maxx)
//        {
//            flag = false;
//            break;
//        }
//    }
//    for (ll i = 1; i <= m; i++)
//    {
//        if (suml[i] != maxx)
//        {
//            flag = false;
//            break;
//        }
//    }
//    if (!flag)
//        ss();
//    else
//    {
//        for (ll i = 1; i <= n; i++)
//        {
//            for (ll j = 1; j <= m; j++)
//                cout << maps[i][j] << " ";
//            cout << endl;
//        }
//    }
//}
//void ss()
//{
//    int maxx = 0;
//    int ff = 0;
//    while (!qq.empty())
//    {
//        auto p = qq.front();
//        qq.pop();
//        if (ff < s.length())
//        {
//            if (s[ff] == 'D')
//            {
//                mapss[p.first][p.second] = maxx - sumrr[p.first];
//                sumrr[p.first] = maxx;
//                sumll[p.second] += mapss[p.first][p.second];
//            }
//            else if (s[ff] == 'R')
//            {
//                mapss[p.first][p.second] = maxx - sumll[p.second];
//                sumll[p.second] = maxx;
//                sumrr[p.first] += mapss[p.first][p.second];
//            }
//            ff++;
//        }
//        else
//        {
//            mapss[p.first][p.second] = maxx - sumll[p.second];
//            sumll[p.second] = maxx;
//            sumrr[p.first] += mapss[p.first][p.second];
//        }
//    }
//    for (ll i = 1; i <= n; i++)
//    {
//        for (ll j = 1; j <= m; j++)
//        {
//            cout << mapss[i][j] << " ";
//        }
//        cout << endl;
//    }
//}
//int main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    ll T;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}